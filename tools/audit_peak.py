"""针对评审 P0-3「削峰」的补充实证：削尾到底影响什么。

三组实验：
  1. 同一时间切分，目标分别用「清洗后（缩尾）」与「原始」——看缩尾是否抬高了测试指标；
  2. 双 11 专项留出：只用 2025-10-31 之前的数据训练，预测 2025 年 11 月——看模型在真实峰值上的表现；
  3. 在上述基础上把 dow 改为 one-hot，看编码修复能挽回多少。

输出：outputs/metrics/review_peak_audit.json
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
from common import DATA_PROC, DATA_RAW, METRICS, save_json  # noqa: E402

BASE = ["is_weekend", "is_holiday", "promo_index", "temp_c", "rain_mm",
        "lag1_volume", "lag7_volume", "roll7_mean"]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else float("nan")
    mape = float(np.mean(np.abs((y - p) / y)) * 100)
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4), "MAPE": round(mape, 2)}


def fit(df: pd.DataFrame, cols: list[str], idx_tr, idx_te, target: str):
    X = df[cols].to_numpy(float)
    y = df[target].to_numpy(float)
    m = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(X[idx_tr], y[idx_tr])
    return m.predict(X[idx_te]), y[idx_te]


def main() -> None:
    out: dict = {}
    proc = pd.read_csv(DATA_PROC / "volume_clean.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
    proc["volume_orig"] = proc["volume_raw"]
    split = int(len(proc) * 0.8)
    idx_tr, idx_te = np.arange(split), np.arange(split, len(proc))

    # 1) 缩尾 vs 原始（同一时间切分）
    fixed = {}
    for target, tag in (("volume", "缩尾目标"), ("volume_orig", "原始目标")):
        p, y = fit(proc, BASE + ["dow", "month"], idx_tr, idx_te, target)
        fixed[tag] = metrics(y, p)
    out["split_compare"] = fixed

    # 2) 双 11 专项留出（含 one-hot 对照）
    raw = pd.read_csv(DATA_RAW / "daily_volume.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
    raw = raw.reset_index(drop=True)
    for c in ("lag1_volume", "lag7_volume", "roll7_mean"):
        if c not in raw:
            pass
    raw["lag1_volume"] = raw["volume"].shift(1)
    raw["lag7_volume"] = raw["volume"].shift(7)
    raw["roll7_mean"] = raw["volume"].shift(1).rolling(7).mean()
    raw = raw.dropna().reset_index(drop=True)

    tr = raw["date"] <= "2025-10-31"
    te = (raw["date"] >= "2025-11-01") & (raw["date"] <= "2025-11-30")
    peak = {}
    p, y = fit(raw, BASE + ["dow", "month"], np.where(tr)[0], np.where(te)[0], "volume")
    peak["数值编码"] = metrics(y, p)
    oh = pd.concat([raw[BASE], pd.get_dummies(raw["dow"], prefix="dow", dtype=float),
                    pd.get_dummies(raw["month"], prefix="mon", dtype=float)], axis=1)
    raw_oh = pd.concat([raw[["date", "volume"]], oh], axis=1)
    p2, y2 = fit(raw_oh, [c for c in raw_oh.columns if c not in ("date", "volume")],
                 np.where(tr)[0], np.where(te)[0], "volume")
    peak["one-hot 编码"] = metrics(y2, p2)

    test_df = raw[te][["date", "volume", "promo_index"]].copy()
    test_df["pred"] = np.round(p, 1)
    test_df["date"] = test_df["date"].dt.strftime("%Y-%m-%d")
    out["peak_holdout"] = {
        "train_end": "2025-10-31", "test_range": "2025-11-01 ~ 2025-11-30",
        "models": peak,
        "worst": test_df.assign(err=(test_df["volume"] - test_df["pred"]).abs())
                        .nlargest(3, "err").round(1).to_dict(orient="records"),
        "mean_actual": round(float(test_df["volume"].mean()), 1),
        "mean_pred": round(float(test_df["pred"].mean()), 1),
    }

    # 3) 训练窗口内缩尾对促销系数的影响
    coef = {}
    for target in ("volume", "volume_raw"):
        X = proc[BASE + ["dow", "month"]].to_numpy(float)
        m = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(X[:split], proc[target][:split])
        coef[target] = round(float(m.named_steps["lr"].coef_[2]), 2)  # promo_index 的系数（原始尺度，需除以缩放）
    out["promo_coef_raw_unit"] = coef

    save_json(out, METRICS / "review_peak_audit.json")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
