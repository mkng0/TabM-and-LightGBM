from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 这样能正确获取到项目根目录 tree-model-learning，再去对接 processed
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

train_df = pd.read_csv(DEFAULT_DIR / "train_processed.csv")
test_df = pd.read_csv(DEFAULT_DIR / "test_processed.csv")

TARGET_COL = "Survived" 
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])

# stratify=y 可以保证训练集和验证集里各类别的比例与原数据集一致（适合分类任务）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集样本量: {X_train.shape[0]}, 验证集样本量: {X_val.shape[0]}")

# 配置随机森林与网格搜索参数
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# 调优的参数组合
param_grid = {
    'n_estimators': [50, 100, 200],       # 树的数量
    'max_depth': [None, 10, 20],      # 树的最大深度
    'min_samples_split': [5, 7, 9],      # 分裂内部节点所需的最小样本数
    'criterion': ['gini', 'entropy']  # 评估节点分裂质量的指标
}

# 运行网格搜索（使用 5 折交叉验证，n_jobs=-1 会用尽你电脑的所有 CPU 核心加速计算）
print("正在进行网格搜索调参，请稍候...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-2)
grid_search.fit(X_train, y_train)

# 输出模型评估结果
print("\n=== 模型训练完成 ===")
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"交叉验证最高准确率: {grid_search.best_score_:.4f}")

# 使用最优的模型在本地验证集上做测试
best_model = grid_search.best_estimator_
val_accuracy = best_model.score(X_val, y_val)
print(f"本地验证集准确率: {val_accuracy:.4f}")

# 预测测试集并本地对比评估
print("正在对测试集进行预测...")
test_predictions = best_model.predict(X_test)

# 只有当 test_df 里面包含真实的 TARGET_COL 时，才能执行下面这段代码：
if TARGET_COL in test_df.columns:
    y_test_true = test_df[TARGET_COL]
    
    test_accuracy = best_model.score(X_test, y_test_true)
    print(f"\n最终测试集泛化准确率: {test_accuracy:.4f}")
    
    print("\n测试集详细评估报告:")
    print(classification_report(y_test_true, test_predictions))

# 将预测结果保存到 processed 文件夹下
output_df = pd.DataFrame({
    "Index": test_df.index, 
    "Prediction": test_predictions
})
output_path = DEFAULT_DIR / "test_predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"预测结果已成功保存至: {output_path}")