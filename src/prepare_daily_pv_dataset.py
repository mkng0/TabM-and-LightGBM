from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only without pandas.
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "daily_pv_debug.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_KEY = "data"
NUMERIC_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "ma_ratio",
    "turnover",
    "turnover_20d",
    "volume_ratio",
]
CAT_FEATURES = ["industry", "market", "size_group"]


def require_pandas() -> None:
    if pd is None:
        raise SystemExit(
            "Missing dependency: pandas. Install pandas and PyTables first, for example:\n"
            "  pip install pandas tables"
        ) from _PANDAS_IMPORT_ERROR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read daily_pv_debug.h5, build labels/features, and split by time."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"HDF5 file path. Defaults to {DEFAULT_DATA_PATH}.",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"HDF5 key to read. Defaults to {DEFAULT_KEY!r}.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print from the top of the dataframe.",
    )
    parser.add_argument(
        "--train-end",
        default="2021-01-01",
        help="Rows before this date are used for training.",
    )
    parser.add_argument(
        "--valid-end",
        default="2022-01-01",
        help="Rows from train-end to before this date are used for validation.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with missing rolling features or labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for processed parquet files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write processed parquet files.",
    )
    return parser.parse_args(argv)


def read_daily_pv(path: Path, key: str):
    require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    try:
        return pd.read_hdf(path, key=key)
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: PyTables. Install it first, for example:\n"
            "  pip install tables"
        ) from exc


def normalize_panel_dataframe(df):
    require_pandas()

    if "date" not in df.columns or "stock_code" not in df.columns:
        df = df.reset_index()

    df = df.rename(
        columns={
            column: column[1:]
            for column in df.columns
            if isinstance(column, str) and column.startswith("$")
        }
    )

    rename_map = {}
    date_candidates = ["date", "datetime", "trade_date", "time"]
    stock_candidates = ["stock_code", "instrument", "symbol", "code", "ticker"]

    if "date" not in df.columns:
        for column in date_candidates:
            if column in df.columns:
                rename_map[column] = "date"
                break

    if "stock_code" not in df.columns:
        for column in stock_candidates:
            if column in df.columns:
                rename_map[column] = "stock_code"
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"date", "stock_code", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if "turnover" not in df.columns:
        if "amount" in df.columns:
            df["turnover"] = pd.to_numeric(df["amount"], errors="coerce")
        else:
            df["turnover"] = df["volume"]
    else:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

    if "market" not in df.columns:
        df["market"] = df["stock_code"].str.extract(r"^([A-Za-z]+)", expand=False)
        df["market"] = df["market"].fillna("unknown")

    for column in ["industry", "size_group"]:
        if column not in df.columns:
            df[column] = "unknown"

    return df


def build_features(df):
    df = normalize_panel_dataframe(df)
    df = df.sort_values(["stock_code", "date"]).copy()

    g = df.groupby("stock_code", group_keys=False)

    df["future_ret"] = g["close"].shift(-1) / df["close"] - 1
    df["median_future_ret"] = df.groupby("date")["future_ret"].transform("median")
    df["y"] = (df["future_ret"] > df["median_future_ret"]).astype("int64")

    df["ret_1d"] = g["close"].pct_change(1, fill_method=None)
    df["ret_5d"] = g["close"].pct_change(5, fill_method=None)
    df["ret_20d"] = g["close"].pct_change(20, fill_method=None)

    df["vol_20d"] = (
        g["ret_1d"].rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    )
    df["ma_5"] = (
        g["close"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    )
    df["ma_20"] = (
        g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    df["ma_ratio"] = df["ma_5"] / df["ma_20"] - 1
    df["turnover_20d"] = (
        g["turnover"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["volume_ratio"] = df["volume"] / (
        g["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    )

    return df


def drop_model_na(df):
    required_columns = NUMERIC_FEATURES + ["future_ret", "median_future_ret", "y"]
    return df.dropna(subset=required_columns).reset_index(drop=True)


def split_by_time(df, train_end: str = "2021-01-01", valid_end: str = "2022-01-01"):
    require_pandas()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)

    train_df = df[df["date"] < train_end_ts].copy()
    valid_df = df[(df["date"] >= train_end_ts) & (df["date"] < valid_end_ts)].copy()
    test_df = df[df["date"] >= valid_end_ts].copy()
    return train_df, valid_df, test_df


def save_processed_data(output_dir: Path, model_df, train_df, valid_df, test_df):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": output_dir / "daily_pv_debug_features.parquet",
        "train": output_dir / "train.parquet",
        "valid": output_dir / "valid.parquet",
        "test": output_dir / "test.parquet",
    }

    try:
        model_df.to_parquet(paths["features"], index=False)
        train_df.to_parquet(paths["train"], index=False)
        valid_df.to_parquet(paths["valid"], index=False)
        test_df.to_parquet(paths["test"], index=False)
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow or fastparquet. Install one first, for example:\n"
            "  pip install pyarrow"
        ) from exc

    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raw_df = read_daily_pv(args.path, args.key)
    featured_df = build_features(raw_df)
    model_df = featured_df if args.keep_na else drop_model_na(featured_df)
    train_df, valid_df, test_df = split_by_time(
        model_df, train_end=args.train_end, valid_end=args.valid_end
    )
    saved_paths = None
    if not args.no_save:
        saved_paths = save_processed_data(
            output_dir=args.output_dir,
            model_df=model_df,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

    print(f"path: {args.path}")
    print(f"key: {args.key}")
    print(f"raw shape: {raw_df.shape}")
    print(f"featured shape: {featured_df.shape}")
    print(f"model shape: {model_df.shape}")
    print(f"numeric features: {NUMERIC_FEATURES}")
    print(f"categorical features: {CAT_FEATURES}")
    print(
        "split sizes: "
        f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
    )
    if saved_paths is not None:
        print("saved files:")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")
    print()
    print(model_df.head(args.head).to_string())


if __name__ == "__main__":
    main()
