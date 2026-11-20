"""在当前数据状态下，重算"修复前→修复后"的逐级对照，供文档引用。

避免文档混用不同数据状态的数字（旧版 36.01 来自缩尾后的数据）。
输出：outputs/metrics/review_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA_PROC, METRICS, save_json  # noqa: E402

# 复现"修复前的缩尾口径"必须用冻结的历史常量，不能按当前数据重算：
# 585.0 是 2026-11-13 那版数据上 IQR 上界（588.8）截断后的实际封顶值。
# 若改为动态推导，对照的就不是"修复前"，before/after 表会失去意义。
CAP_LEGACY_IQR = 585.0


def met(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 2),
        "R2": round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "MAPE": round(float(np.mean(np.abs((y - p) / y)) * 100), 2),
    }


def sin_cos(df: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    X = df[base].copy()
    X["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    X["month"] = df["month"]
    return X


def main() -> None:
    df = pd.read_csv(DATA_PROC / "volume_clean.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    split = int(len(df) * 0.8)
    y = df["volume"].to_numpy(float)

    base = ["is_weekend", "is_holiday", "promo_index", "temp_c", "rain_mm",
            "lag1_volume", "lag7_volume", "roll7_mean"]

    def fit(X: pd.DataFrame, target: np.ndarray) -> dict:
        m = Pipeline([("s", StandardScaler()), ("l", LinearRegression())]).fit(X.iloc[:split], target[:split])
        return met(target[split:], m.predict(X.iloc[split:]))

    # 1) 修复前：数值编码、无校历、目标缩尾
    X_old = df[base].copy(); X_old["dow"] = df["dow"]; X_old["month"] = df["month"]
    r_clip = fit(X_old, np.minimum(y, CAP_LEGACY_IQR))
    r_old = fit(X_old, y)

    # 1b) 正余弦编码（仍无校历）
    r_sin = fit(sin_cos(df, base), y)

    # 2) dow one-hot（仍无校历）
    X_dow = df[base].copy(); X_dow["month"] = df["month"]
    for i in range(1, 7):
        X_dow[f"dow_{i}"] = (df["dow"] == i).astype(float)
    r_dow = fit(X_dow, y)

    # 3) + 校历（month 数值）——当前线上配置
    X_full = X_dow.copy(); X_full["is_vacation"] = df["is_vacation"]
    r_full = fit(X_full, y)

    # 4) + month one-hot
    X_moh = X_full.drop(columns=["month"]).copy()
    for m in range(2, 13):
        X_moh[f"mon_{m}"] = (df["month"] == m).astype(float)
    r_moh = fit(X_moh, y)

    out = {
        "data_state": "volume_clean.csv（499 条工单版本，目标未缩尾）",
        "n_train": split, "n_test": len(df) - split,
        "ladder": {
            "原做法（数值编码 + 缩尾目标）": r_clip,
            "数值编码 + 未缩尾目标": r_old,
            "正余弦编码": r_sin,
            "+ dow one-hot": r_dow,
            "+ is_vacation（线上配置）": r_full,
            "+ month one-hot": r_moh,
        },
    }
    save_json(out, METRICS / "review_ablation.json")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
