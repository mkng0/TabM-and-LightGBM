# 基于 TabM 与 LightGBM 集成的股票收益方向预测项目实现指南

## 1. 项目目标

本项目目标是构建一个面向秋招机器学习岗位的完整项目：

> 基于 A 股历史行情和因子特征，预测股票下一期收益方向，并对比 LightGBM、CatBoost、TabM 等模型，最终构建 LightGBM + TabM 集成模型。

项目重点不是直接预测股价，而是预测：

```text
某只股票在下一期是否上涨
或
某只股票在下一期是否跑赢市场中位数收益
```

## 2. 数据形式

原始数据通常是面板数据：

```text
时间 × 股票 × 特征
```

例如：

| date | stock_code | close | volume | turnover | industry |
|---|---|---:|---:|---:|---|
| 2020-01-01 | 000001 | 12.31 | 1000000 | 0.032 | bank |
| 2020-01-01 | 000002 | 8.42 | 800000 | 0.021 | real_estate |
| 2020-01-02 | 000001 | 12.48 | 1200000 | 0.041 | bank |

TabM 不能直接输入三维张量：

```python
x.shape = (time, stock, feature)
```

需要展开成二维表格：

```text
每一行 = 某一天的某只股票
每一列 = 一个特征
```

最终输入形状为：

```python
x_num.shape = (样本数, 数值特征数)
x_cat.shape = (样本数, 类别特征数)
y.shape = (样本数,)
```

如果有 500 个交易日、200 只股票、30 个特征，则展开后：

```text
样本数 = 500 × 200 = 100000
特征数 = 30
```

## 3. 推荐项目结构

```text
stock-ml-tabm-lgbm/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_tabm_experiment.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_lgbm.py
│   │   ├── train_catboost.py
│   │   ├── train_tabm.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── backtest.py
│   └── serving/
│       └── app.py
├── configs/
│   ├── lgbm.yaml
│   └── tabm.yaml
├── outputs/
│   ├── figures/
│   ├── models/
│   └── metrics/
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 4. 标签构造

### 4.1 方向预测标签

预测下一期是否上涨：

```python
df["future_ret"] = df.groupby("stock_code")["close"].pct_change().shift(-1)
df["y"] = (df["future_ret"] > 0).astype(int)
```

### 4.2 跑赢市场标签

预测下一期是否跑赢当期横截面中位数：

```python
df["future_ret"] = df.groupby("stock_code")["close"].pct_change().shift(-1)
df["median_future_ret"] = df.groupby("date")["future_ret"].transform("median")
df["y"] = (df["future_ret"] > df["median_future_ret"]).astype(int)
```

更推荐第二种，因为金融市场整体上涨或下跌时，单纯预测正负收益容易受市场行情影响。

## 5. 特征工程

建议先做滚动特征，把时间序列信息压缩成表格特征。

```python
def build_features(df):
    df = df.sort_values(["stock_code", "date"]).copy()

    g = df.groupby("stock_code")

    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_20d"] = g["close"].pct_change(20)

    df["vol_20d"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    df["ma_5"] = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["ma_20"] = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["ma_ratio"] = df["ma_5"] / df["ma_20"] - 1

    df["turnover_20d"] = (
        g["turnover"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["volume_ratio"] = df["volume"] / (
        g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    )

    return df
```

推荐数值特征：

```text
ret_1d
ret_5d
ret_20d
vol_20d
ma_ratio
turnover
turnover_20d
volume_ratio
```

推荐类别特征：

```text
industry
market
size_group
```

不建议一开始加入 `stock_code`，因为它容易让模型记住个股 ID，导致过拟合。后续可以作为对比实验加入。

## 6. 时间切分

金融数据不能随机切分，必须按照时间切分。

```python
train_df = df[df["date"] < "2021-01-01"]
valid_df = df[(df["date"] >= "2021-01-01") & (df["date"] < "2022-01-01")]
test_df = df[df["date"] >= "2022-01-01"]
```

推荐在 README 中明确写：

```text
本项目采用时间序列切分，避免未来信息泄露。
```

## 7. TabM 输入处理

TabM 的输入分为两部分：

```text
x_num: 数值特征，float32
x_cat: 类别特征，long/int64
```

### 7.1 数值特征处理

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_cols = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "ma_ratio",
    "turnover",
    "turnover_20d",
    "volume_ratio",
]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_num = imputer.fit_transform(train_df[num_cols])
X_valid_num = imputer.transform(valid_df[num_cols])
X_test_num = imputer.transform(test_df[num_cols])

X_train_num = scaler.fit_transform(X_train_num)
X_valid_num = scaler.transform(X_valid_num)
X_test_num = scaler.transform(X_test_num)
```

### 7.2 类别特征处理

类别变量需要编码为从 0 开始的连续整数。

```python
from sklearn.preprocessing import OrdinalEncoder

cat_cols = ["industry", "market", "size_group"]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_cat = encoder.fit_transform(train_df[cat_cols].fillna("missing"))
X_valid_cat = encoder.transform(valid_df[cat_cols].fillna("missing"))
X_test_cat = encoder.transform(test_df[cat_cols].fillna("missing"))
```

注意：TabM 的类别取值必须在：

```text
0, 1, 2, ..., cardinality - 1
```

所以如果用了 `unknown_value=-1`，需要把它平移到非负整数：

```python
X_train_cat = X_train_cat.astype("int64") + 1
X_valid_cat = X_valid_cat.astype("int64") + 1
X_test_cat = X_test_cat.astype("int64") + 1

cat_cardinalities = [
    int(max(X_train_cat[:, i].max(), X_valid_cat[:, i].max(), X_test_cat[:, i].max()) + 1)
    for i in range(X_train_cat.shape[1])
]
```

### 7.3 转成 PyTorch 张量

```python
import torch

x_train_num = torch.tensor(X_train_num, dtype=torch.float32)
x_valid_num = torch.tensor(X_valid_num, dtype=torch.float32)
x_test_num = torch.tensor(X_test_num, dtype=torch.float32)

x_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
x_valid_cat = torch.tensor(X_valid_cat, dtype=torch.long)
x_test_cat = torch.tensor(X_test_cat, dtype=torch.long)

y_train = torch.tensor(train_df["y"].values, dtype=torch.float32)
y_valid = torch.tensor(valid_df["y"].values, dtype=torch.float32)
y_test = torch.tensor(test_df["y"].values, dtype=torch.float32)
```

## 8. TabM 模型训练

安装：

```bash
pip install tabm
```

基础模型：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tabm import TabM

n_num_features = x_train_num.shape[1]
d_out = 1

model = TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
loss_fn = nn.BCEWithLogitsLoss()

train_ds = TensorDataset(x_train_num, x_train_cat, y_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

for epoch in range(30):
    model.train()
    total_loss = 0.0

    for xb_num, xb_cat, yb in train_loader:
        optimizer.zero_grad()

        pred = model(xb_num, xb_cat)

        # TabM 输出形状为 (batch_size, k, d_out)
        # 对 k 个 ensemble 预测取平均
        pred = pred.mean(dim=1).squeeze(-1)

        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)

    print(f"epoch={epoch}, loss={total_loss / len(train_ds):.6f}")
```

## 9. LightGBM 强基线

```python
import lightgbm as lgb

X_train_lgbm = train_df[num_cols + cat_cols].copy()
X_valid_lgbm = valid_df[num_cols + cat_cols].copy()

for col in cat_cols:
    X_train_lgbm[col] = X_train_lgbm[col].astype("category")
    X_valid_lgbm[col] = X_valid_lgbm[col].astype("category")

model_lgbm = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model_lgbm.fit(
    X_train_lgbm,
    train_df["y"],
    eval_set=[(X_valid_lgbm, valid_df["y"])],
    eval_metric="auc",
    categorical_feature=cat_cols,
)
```

## 10. 模型评估

分类指标：

```python
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def evaluate_classification(y_true, pred_prob):
    pred_label = (pred_prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, pred_prob),
        "accuracy": accuracy_score(y_true, pred_label),
        "f1": f1_score(y_true, pred_label),
    }
```

金融指标：

```python
def evaluate_rank_ic(df, pred_col="pred_prob", ret_col="future_ret"):
    ic_by_date = (
        df.groupby("date")
        .apply(lambda x: x[pred_col].corr(x[ret_col], method="spearman"))
        .dropna()
    )
    return {
        "rank_ic_mean": ic_by_date.mean(),
        "rank_ic_std": ic_by_date.std(),
        "rank_ic_ir": ic_by_date.mean() / ic_by_date.std(),
    }
```

## 11. LightGBM + TabM 集成

简单加权平均：

```python
final_pred = 0.6 * pred_lgbm + 0.4 * pred_tabm
```

可以在验证集上搜索最优权重：

```python
best_auc = -1
best_w = None

for w in [i / 10 for i in range(11)]:
    pred = w * pred_lgbm_valid + (1 - w) * pred_tabm_valid
    auc = roc_auc_score(y_valid, pred)

    if auc > best_auc:
        best_auc = auc
        best_w = w

print("best weight for LightGBM:", best_w)
```

## 12. 简单回测

每个交易日选择预测概率最高的前 20% 股票，等权买入。

```python
def top_quantile_backtest(df, pred_col="pred_prob", ret_col="future_ret", q=0.8):
    df = df.copy()

    def select_top(x):
        threshold = x[pred_col].quantile(q)
        x["selected"] = (x[pred_col] >= threshold).astype(int)
        return x

    df = df.groupby("date", group_keys=False).apply(select_top)

    daily_ret = (
        df[df["selected"] == 1]
        .groupby("date")[ret_col]
        .mean()
        .dropna()
    )

    cumulative_ret = (1 + daily_ret).cumprod()

    sharpe = daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
    max_drawdown = (cumulative_ret / cumulative_ret.cummax() - 1).min()

    return {
        "daily_ret": daily_ret,
        "cumulative_ret": cumulative_ret,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
```

## 13. FastAPI 部署接口

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StockFeatures(BaseModel):
    ret_1d: float
    ret_5d: float
    ret_20d: float
    vol_20d: float
    ma_ratio: float
    turnover: float
    turnover_20d: float
    volume_ratio: float
    industry: str
    market: str
    size_group: str

@app.post("/predict")
def predict(features: StockFeatures):
    # 1. 转 DataFrame
    # 2. 数值特征 imputer + scaler
    # 3. 类别特征 encoder
    # 4. LightGBM + TabM 预测
    # 5. 加权平均
    return {
        "prob_up": 0.63,
        "label": 1,
        "model": "lgbm_tabm_ensemble_v1"
    }
```

启动：

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

## 14. 简历写法

可以写成：

> 基于 A 股历史行情与因子特征，构建股票收益方向预测系统。项目采用时间序列切分避免未来信息泄露，对比 Logistic Regression、Random Forest、LightGBM、CatBoost、TabM 等模型，并设计 LightGBM + TabM 加权集成模型提升预测稳定性。使用 SHAP 解释模型特征贡献，并基于预测概率构建 Top-K 选股回测策略，最终通过 FastAPI 与 Docker 实现模型服务化部署。

## 15. 面试可讲亮点

1. 面板数据被展开为 `(date, stock)` 级别样本，符合 TabM 的二维表格输入要求。
2. 使用时间序列切分而不是随机切分，避免未来信息泄露。
3. LightGBM 作为工业强基线，TabM 作为先进表格深度模型。
4. 不只看 AUC，还加入 RankIC、Sharpe Ratio、最大回撤等金融指标。
5. 使用模型集成提升预测稳定性。
6. 使用 FastAPI + Docker 做模型服务化，体现机器学习工程能力。

## 16. 推荐实现顺序

```text
第 1 步：整理数据，构造 (date, stock) 样本
第 2 步：构造 y 标签和滚动特征
第 3 步：训练 Logistic Regression / Random Forest
第 4 步：训练 LightGBM / CatBoost
第 5 步：训练 TabM
第 6 步：做 LightGBM + TabM 集成
第 7 步：加入 RankIC 和 Top-K 回测
第 8 步：加入 SHAP 解释
第 9 步：用 FastAPI 封装预测接口
第 10 步：整理 README 和简历描述
```
