from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


# 与 randomforest.py 保持一致：从项目根目录下的 data/processed 读取数据。
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TARGET_COL = "Survived"
DEFAULT_OUTPUT_PATH = DEFAULT_DIR / "test_predictions_lgbm.csv"


def _is_same_path(path_entry: str, target: Path) -> bool:
    try:
        return Path(path_entry or ".").resolve() == target
    except OSError:
        return False


@contextmanager
def _without_script_directory_on_path():
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    sys.path[:] = [
        path_entry
        for path_entry in sys.path
        if not _is_same_path(path_entry, script_dir)
    ]

    try:
        yield
    finally:
        sys.path[:] = original_path


def _load_lgbm_classifier():
    module_name = "lightgbm"
    script_path = Path(__file__).resolve()
    shadow_module = sys.modules.get(module_name)

    if shadow_module is not None:
        module_file = getattr(shadow_module, "__file__", None)
        if module_file is not None and Path(module_file).resolve() == script_path:
            del sys.modules[module_name]

    try:
        with _without_script_directory_on_path():
            module = importlib.import_module(module_name)
            return module.LGBMClassifier
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "缺少 lightgbm 依赖，请先运行: pip install lightgbm"
        ) from exc


def load_processed_data(
    data_dir: str | Path = DEFAULT_DIR,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series | None]:
    """Load processed train/test CSV files using the same layout as randomforest.py."""
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "train_processed.csv", encoding="utf-8")
    test_df = pd.read_csv(data_path / "test_processed.csv", encoding="utf-8")

    if target_col not in train_df.columns:
        raise ValueError(f"训练数据缺少目标列: {target_col}")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
    else:
        X_test = test_df.copy()
        y_test = None

    return X, y, X_test, y_test


def build_lgbm_classifier(random_state: int = 42):
    """Create a binary LightGBM classifier."""
    LGBMClassifier = _load_lgbm_classifier()

    return LGBMClassifier(
        objective="binary",
        random_state=random_state,
        class_weight="balanced",
        verbosity=-1,
    )


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_jobs: int = -2,
    param_grid: dict | None = None,
):
    """Train LightGBM with a compact GridSearchCV parameter grid."""
    lgbm = build_lgbm_classifier(random_state=42)
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "min_child_samples": [10, 20],
        }

    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )
    with _without_script_directory_on_path():
        grid_search.fit(X_train, y_train)
    return grid_search


def save_predictions(
    predictions: Iterable[int],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save model predictions to a UTF-8 CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_values = list(predictions)
    output_df = pd.DataFrame(
        {
            "Index": range(len(prediction_values)),
            "Prediction": prediction_values,
        }
    )
    output_df.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> None:
    X, y, X_test, y_test = load_processed_data()

    # stratify=y 可以保证训练集和验证集里各类别比例与原数据一致。
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"训练集样本量: {X_train.shape[0]}, 验证集样本量: {X_val.shape[0]}")
    print("正在进行 LightGBM 网格搜索调参，请稍候...")
    grid_search = train_lgbm(X_train, y_train)

    print("\n=== LightGBM 模型训练完成 ===")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最高 AUC: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    print(f"本地验证集准确率: {val_accuracy:.4f}")

    print("正在对测试集进行预测...")
    test_predictions = best_model.predict(X_test)

    if y_test is not None:
        test_accuracy = best_model.score(X_test, y_test)
        print(f"\n最终测试集泛化准确率: {test_accuracy:.4f}")
        print("\n测试集详细评估报告:")
        print(classification_report(y_test, test_predictions))

    output_path = save_predictions(test_predictions)
    print(f"预测结果已成功保存至: {output_path}")


if __name__ == "__main__":
    main()
