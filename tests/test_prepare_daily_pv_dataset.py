from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.prepare_daily_pv_dataset import (
    build_features,
    normalize_panel_dataframe,
    save_processed_data,
    split_by_time,
)


class PrepareDailyPvDatasetTest(unittest.TestCase):
    def test_normalize_panel_dataframe_handles_qlib_hdf5_shape(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-01"), "SH000300"),
                (pd.Timestamp("2020-01-02"), "SH000300"),
            ],
            names=["datetime", "instrument"],
        )
        raw = pd.DataFrame(
            {
                "$open": [1.0, 1.1],
                "$close": [1.1, 1.2],
                "$high": [1.2, 1.3],
                "$low": [0.9, 1.0],
                "$volume": [1000.0, 1100.0],
                "$factor": [1.0, 1.0],
            },
            index=index,
        )

        normalized = normalize_panel_dataframe(raw)

        self.assertIn("date", normalized.columns)
        self.assertIn("stock_code", normalized.columns)
        self.assertIn("close", normalized.columns)
        self.assertIn("volume", normalized.columns)
        self.assertEqual(normalized.loc[0, "stock_code"], "SH000300")
        self.assertEqual(normalized.loc[0, "close"], 1.1)

    def test_build_features_creates_target_and_rolling_features(self) -> None:
        dates = pd.date_range("2020-01-01", periods=25, freq="D")
        rows = []
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "stock_code": "A",
                    "close": 100 + i * 2,
                    "volume": 1000 + i,
                }
            )
            rows.append(
                {
                    "date": date,
                    "stock_code": "B",
                    "close": 100 + i,
                    "volume": 2000 + i,
                }
            )

        featured = build_features(pd.DataFrame(rows))

        first_day = featured[featured["date"] == dates[0]].set_index("stock_code")
        self.assertEqual(first_day.loc["A", "y"], 1)
        self.assertEqual(first_day.loc["B", "y"], 0)

        for column in [
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "ma_ratio",
            "turnover",
            "turnover_20d",
            "volume_ratio",
            "industry",
            "market",
            "size_group",
        ]:
            self.assertIn(column, featured.columns)

        day_21 = featured[featured["date"] == dates[20]]
        self.assertTrue(day_21["ret_20d"].notna().all())
        self.assertTrue(day_21["vol_20d"].notna().all())

    def test_build_features_derives_market_and_placeholder_categories(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="D"),
                "stock_code": ["SH000300", "SH000300"],
                "close": [1.0, 1.1],
                "volume": [1000.0, 1100.0],
            }
        )

        featured = build_features(df)

        self.assertEqual(featured.loc[0, "market"], "SH")
        self.assertEqual(featured.loc[0, "industry"], "unknown")
        self.assertEqual(featured.loc[0, "size_group"], "unknown")

    def test_split_by_time_uses_chronological_boundaries(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-12-31", "2021-01-01", "2022-01-01"]),
                "stock_code": ["A", "A", "A"],
                "close": [1.0, 2.0, 3.0],
            }
        )

        train_df, valid_df, test_df = split_by_time(
            df, train_end="2021-01-01", valid_end="2022-01-01"
        )

        self.assertEqual(len(train_df), 1)
        self.assertEqual(len(valid_df), 1)
        self.assertEqual(len(test_df), 1)

    def test_save_processed_data_writes_feature_and_split_files(self) -> None:
        model_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-12-31", "2021-01-01", "2022-01-01"]),
                "stock_code": ["A", "B", "C"],
                "y": [1, 0, 1],
            }
        )
        train_df = model_df.iloc[[0]].copy()
        valid_df = model_df.iloc[[1]].copy()
        test_df = model_df.iloc[[2]].copy()

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            paths = save_processed_data(
                output_dir=output_dir,
                model_df=model_df,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
            )

            self.assertEqual(
                set(paths),
                {"features", "train", "valid", "test"},
            )
            for path in paths.values():
                self.assertTrue(path.exists())

            saved_train = pd.read_parquet(paths["train"])
            self.assertEqual(len(saved_train), 1)
            self.assertEqual(saved_train.loc[0, "stock_code"], "A")


if __name__ == "__main__":
    unittest.main()
