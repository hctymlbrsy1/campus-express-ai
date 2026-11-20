"""峰值留出的口径复核（回应评审对"0.666 是诚实答案"的质疑）。

两个留出集：
  A. 同级别峰值：训练 ≤2026-05-31，测试 2026-06-01~06-20（618，promo=0.65，
     训练集含 2025 年 618，月份 6 也出现过）——不外推；
  B. 外推压力：训练 ≤2025-10-31，测试 2025-11-01~11-30（双 11，promo=1.0，
     月份 11 未出现）——promo 与 month 双重外推。

每个留出集跑 2×3 矩阵：目标（原始/缩尾）× 编码（数值 / one-hot / one-hot+is_vacation）。
输出：outputs/metrics/peak_matrix.json
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
CLIP_CAP = 585.0


def met(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
        "MAE": round(float(np.mean(np.abs(y - p))), 2),
        "R2": round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
        "MAPE": round(float(np.mean(np.abs((y - p) / y)) * 100), 2),
        "mean_pred": round(float(p.mean()), 1),
        "mean_true": round(float(y.mean()), 1),
    }


def frame(target: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA_RAW / "daily_volume.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
    raw = raw.reset_index(drop=True)
    raw["volume_orig"] = raw["volume"]
    raw["volume_clip"] = raw["volume"].clip(upper=CLIP_CAP)
    # 滞后一律用同一口径重建，避免混用
    for c, src in (("lag1_volume", 1), ("lag7_volume", 7)):
        raw[c] = raw[target].shift(src)
    raw["roll7_mean"] = raw[target].shift(1).rolling(7).mean()
    return raw.dropna().reset_index(drop=True)


def run(df: pd.DataFrame, tr_mask, te_mask, enc: str, target: str) -> dict:
    feats = list(BASE)
    X = df[feats].copy()
    if enc == "numeric":
        X["dow"] = df["dow"]; X["month"] = df["month"]
    else:
        X = pd.concat([X, pd.get_dummies(df["dow"], prefix="dow", dtype=float)], axis=1)
        if enc in ("onehot_vac", "onehot_vac_nummonth"):
            X = pd.concat([X, df[["is_vacation"]]], axis=1)
            if enc == "onehot_vac":
                X = pd.concat([X, pd.get_dummies(df["month"], prefix="mon", dtype=float)], axis=1)
            else:
                X["month"] = df["month"]
        else:
            X["month"] = df["month"]
    X = X.astype(float)
    m = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
    m.fit(X[tr_mask], df[target][tr_mask])
    return met(df[target][te_mask], m.predict(X[te_mask]))


def main() -> None:
    out: dict = {}
    for tag, (target, tr_end, te_start, te_end) in {
        "same_level_618": ("volume", "2026-05-31", "2026-06-01", "2026-06-20"),
        "extrap_1111": ("volume", "2025-10-31", "2025-11-01", "2025-11-30"),
    }.items():
        block = {}
        for tgt in ("volume_orig", "volume_clip"):
            df = frame(tgt)
            tr = df["date"] <= tr_end
            te = (df["date"] >= te_start) & (df["date"] <= te_end)
            block[tgt] = {enc: run(df, tr, te, enc, tgt) for enc in ("numeric", "onehot", "onehot_vac_nummonth", "onehot_vac")}
        out[tag] = block

    save_json(out, METRICS / "peak_matrix.json")
    for tag, block in out.items():
        print(f"\n=== {tag} ===")
        for tgt, rows in block.items():
            for enc, m in rows.items():
                print(f"  target={tgt:11s} enc={enc:11s} RMSE={m['RMSE']:7.2f} R2={m['R2']:7.4f} "
                      f"pred={m['mean_pred']:6.1f} true={m['mean_true']:6.1f}")


if __name__ == "__main__":
    main()
