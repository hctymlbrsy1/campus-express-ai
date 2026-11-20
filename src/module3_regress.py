"""模块三：日到件量预测（回归）

评估设计（回应外部评审的三点质疑）
------------------------------------------------
1. 目标不再缩尾：双 11 峰值是真实信号，IQR 只用于打 `is_outlier` 标记；
2. 特征编码：`dow` 用 one-hot（星期效应非单调），`month` 保留数值，
   新增 `is_vacation`（校历效应）——这是"工程量最大但收益最高"的修复；
3. 三套评估口径并列：
   - 主测试集：后 20% 时间序列，分布内；
   - 618 同级别留出：促销强度 0.65、月份 6 在训练集都出现过，**不外推**；
   - 双 11 外推压力：促销 1.0、月份 11 均未见过，仅作压力测试，并给出逐级修复的阶梯表。

输出：outputs/metrics/module3.json、figures/regression.png、models/volume_regressor.joblib
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA_PROC, FIGURES, MODELS, METRICS, SEED, ensure_dirs, save_json, setup_matplotlib, timer

# 数值特征（含 month 与 is_vacation）
# 注意：不再放 is_weekend —— 它与 dow_5 + dow_6 完全共线（设计阵秩 15/16），
# 会让 PCA 载荷与回归系数不可单独解读。周末信息已由 dow 哑变量完整表达。
NUMERIC = ["is_holiday", "is_vacation", "promo_index", "temp_c", "rain_mm",
           "lag1_volume", "lag7_volume", "roll7_mean", "month"]
DOW_COLS = [f"dow_{i}" for i in range(1, 7)]      # 以周一为基准
FEATURE_ORDER = NUMERIC + DOW_COLS

FEATURE_CN = {
    "dow": "星期", "is_weekend": "是否周末", "is_holiday": "是否节假日", "is_vacation": "是否寒暑假",
    "promo_index": "促销强度", "temp_c": "气温", "rain_mm": "降雨量",
    "lag1_volume": "前1日到件量", "lag7_volume": "前7日到件量",
    "roll7_mean": "近7日均值", "month": "月份",
}
NO_LAG = ["is_holiday", "is_vacation", "promo_index", "temp_c", "rain_mm", "month"] + DOW_COLS
NO_PROMO = [c for c in FEATURE_ORDER if c != "promo_index"]


def design(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """把含 dow 的原始表转成设计矩阵（dow one-hot），供训练与服务端共用。"""
    X = df.copy()
    for i in range(1, 7):
        X[f"dow_{i}"] = (X["dow"] == i).astype(float)
    return X[cols or FEATURE_ORDER].astype(float)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "MAE": round(float(mean_absolute_error(y, p)), 2),
        "R2": round(float(r2_score(y, p)), 4),
        "MAPE": round(float(np.mean(np.abs((y - p) / y)) * 100), 2),
        "mean_pred": round(float(p.mean()), 1),
        "mean_true": round(float(y.mean()), 1),
    }


def build(model: str) -> Pipeline:
    if model == "linear":
        est = LinearRegression()
    else:
        est = MLPRegressor(
            hidden_layer_sizes=(32, 32), activation="relu", alpha=1e-3,
            learning_rate_init=1e-2, max_iter=3000, early_stopping=True,
            n_iter_no_change=50, random_state=SEED,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", est)])


def fit_predict(df: pd.DataFrame, cols: list[str], tr_mask, te_mask, model: str = "linear"):
    X, y = design(df, cols), df["volume"].to_numpy(float)
    pipe = build(model).fit(X[tr_mask], y[tr_mask])
    return pipe.predict(X[te_mask]), y[te_mask]


def peak_ladder(df: pd.DataFrame, tr_mask, te_mask) -> dict:
    """同一留出集上逐级修复：数值 → dow one-hot → +校历 → +月份 one-hot。"""
    variants = {}
    # "原做法"要忠实复现修复前的特征集（其中含 is_weekend）。
    # 现行配置不再放 is_weekend（与 dow 哑变量完全共线），但两者拟合值完全相同，
    # 所以阶梯第 3 行的数字仍是现行模型的等价结果。
    base_num = ["is_weekend", "is_holiday", "promo_index", "temp_c", "rain_mm",
                "lag1_volume", "lag7_volume", "roll7_mean"]
    # 1) 原始做法：dow/month 当数值，无校历
    X1 = df[base_num].copy(); X1["dow"] = df["dow"]; X1["month"] = df["month"]
    # 2) dow one-hot
    X2 = df[base_num].copy(); X2["month"] = df["month"]
    for i in range(1, 7):
        X2[f"dow_{i}"] = (df["dow"] == i).astype(float)
    # 3) + 校历（month 数值）——现行配置
    X3 = X2.copy(); X3["is_vacation"] = df["is_vacation"]
    y = df["volume"].to_numpy(float)
    for tag, X in (("数值编码（原做法）", X1), ("dow one-hot", X2), ("+ 校历 is_vacation", X3)):
        pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(X[tr_mask], y[tr_mask])
        variants[tag] = metrics(y[te_mask], pipe.predict(X[te_mask]))
    return variants


def month_encoding_618(df: pd.DataFrame, tr_mask, te_mask) -> dict:
    """在 618 同级别留出上比较 month 编码。

    双 11 留出的训练期月份只有 6–10，month=11 的哑变量系数恒为 0（不可识别），
    因此 month 编码的对比必须换到这个"月份在训练集出现过"的窗口来做。
    """
    y = df["volume"].to_numpy(float)
    base = design(df).drop(columns=["month"])
    num = base.copy(); num["month"] = df["month"]
    oh = base.copy()
    for m in range(2, 13):
        oh[f"mon_{m}"] = (df["month"] == m).astype(float)
    out = {}
    for tag, X in (("month 数值", num), ("month one-hot", oh)):
        pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(X[tr_mask], y[tr_mask])
        out[tag] = metrics(y[te_mask], pipe.predict(X[te_mask]))
    return out


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PROC / "volume_clean.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    split = int(len(df) * 0.8)
    tr_all, te_all = np.arange(len(df)) < split, np.arange(len(df)) >= split

    # ---------------- 主测试集
    results, preds = {}, {}
    for key, name in (("linear", "线性回归"), ("mlp", "MLP回归")):
        pipe = build(key)
        with timer() as t:
            pipe.fit(design(df)[tr_all], df["volume"].to_numpy(float)[tr_all])
        p = pipe.predict(design(df)[te_all])
        preds[key] = p
        m = metrics(df["volume"].to_numpy(float)[te_all], p)
        m.update({"name": name, "train_ms": round(t["ms"], 1)})
        results[key] = m
    best = min(results, key=lambda k: results[k]["RMSE"])
    best_pipe = build(best).fit(design(df)[tr_all], df["volume"].to_numpy(float)[tr_all])
    joblib.dump({"pipeline": best_pipe, "features": FEATURE_ORDER, "feature_cn": FEATURE_CN,
                 "numeric": NUMERIC, "dow_cols": DOW_COLS}, MODELS / "volume_regressor.joblib")

    # ---------------- 特征消融（主测试集）
    ablation = {}
    for tag, cols in (("完整特征", FEATURE_ORDER), ("去掉滞后特征", NO_LAG), ("去掉促销特征", NO_PROMO)):
        abl = {}
        for key in ("linear", "mlp"):
            p, y = fit_predict(df, cols, tr_all, te_all, key)
            abl[key] = metrics(y, p)["RMSE"]
        abl["best_model"] = min(("linear", "mlp"), key=lambda k: abl[k])
        ablation[tag] = abl

    # ---------------- 618 同级别留出（不外推）
    tr618 = df["date"] <= "2026-05-31"
    te618 = (df["date"] >= "2026-06-01") & (df["date"] <= "2026-06-20")
    p618, y618 = fit_predict(df, FEATURE_ORDER, tr618, te618)
    same_level = {
        "train_end": "2026-05-31", "test_range": "2026-06-01 ~ 2026-06-20", "n": int(te618.sum()),
        "linear": metrics(y618, p618),
        "month_encoding": month_encoding_618(df, tr618, te618),
        "note": "促销强度 0.65 与月份 6 均在训练集中出现过，属同级别峰值；窗口方差小，R² 不具解释力，以 MAPE 为准",
    }

    # ---------------- 双 11 外推压力
    tr1111 = df["date"] <= "2025-10-31"
    te1111 = (df["date"] >= "2025-11-01") & (df["date"] <= "2025-11-30")
    ladder = peak_ladder(df, tr1111, te1111)
    p11, _ = fit_predict(df, FEATURE_ORDER, tr1111, te1111)
    test11 = df[te1111][["date", "volume", "promo_index", "is_vacation"]].copy()
    test11["pred"] = p11
    test11["err"] = (test11["volume"] - test11["pred"]).abs()
    worst11 = test11.nlargest(3, "err")
    worst11_records = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "volume": int(r["volume"]),
            "pred": round(float(r["pred"]), 1),
            "err": round(float(r["err"]), 1),
            "promo_index": float(r["promo_index"]),   # 控制参数不参与四舍五入
            "is_vacation": int(r["is_vacation"]),
        }
        for _, r in worst11.iterrows()
    ]
    extrapolation = {
        "train_end": "2025-10-31", "test_range": "2025-11-01 ~ 2025-11-30", "n": int(te1111.sum()),
        "ladder": ladder,
        "worst": worst11_records,
        "note": "促销 1.0 与月份 11 均未在训练集出现，属双重外推；仅作压力测试，不进主结论。"
                "month one-hot 在此不可识别（mon_11 训练样本为 0、系数恒为 0），故阶梯只保留三级，"
                "月份编码对比改在 618 窗口（见 same_level_peak_618.month_encoding）",
    }

    # ---------------- PCA 影响因素（主训练集拟合）
    Xs = StandardScaler().fit_transform(design(df))
    pca = PCA(n_components=min(len(FEATURE_ORDER), 12), random_state=SEED).fit(Xs)
    Z = pca.transform(Xs)
    corr = [round(float(np.corrcoef(Z[:, i], df["volume"])[0, 1]), 3) for i in range(3)]
    labels = [FEATURE_CN.get(c, c) for c in FEATURE_ORDER]
    loadings = pd.DataFrame(pca.components_[:3].T, index=labels, columns=["PC1", "PC2", "PC3"]).round(3)

    # 共线对照：把 is_weekend 加回去重算一次，量化"秩亏污染了哪一维"
    # （文档里"PC2 13.36% → 8.72%"这类数字必须能在产物中找到，故在此落盘）
    X16 = design(df).copy()
    X16["is_weekend"] = (df["dow"] >= 5).astype(float)
    pca16 = PCA(n_components=3, random_state=SEED).fit(StandardScaler().fit_transform(X16.to_numpy(float)))
    idx_we = list(X16.columns).index("is_weekend")
    collinearity = {
        "rank_16dim": int(np.linalg.matrix_rank(X16.to_numpy(float))),
        "pc1_before": round(float(pca16.explained_variance_ratio_[0]), 4),
        "pc1_after": round(float(pca.explained_variance_ratio_[0]), 4),
        "pc2_before": round(float(pca16.explained_variance_ratio_[1]), 4),
        "pc2_after": round(float(pca.explained_variance_ratio_[1]), 4),
        "is_weekend_pc2_loading": round(float(pca16.components_[1, idx_we]), 3),
        "pc1_before_pct": round(float(pca16.explained_variance_ratio_[0]) * 100, 2),
        "pc1_after_pct": round(float(pca.explained_variance_ratio_[0]) * 100, 2),
        "pc2_before_pct": round(float(pca16.explained_variance_ratio_[1]) * 100, 2),
        "pc2_after_pct": round(float(pca.explained_variance_ratio_[1]) * 100, 2),
    }

    # ---------------- 可视化
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
    dates = df["date"].iloc[split:]
    y_true = df["volume"].iloc[split:]
    axes[0].plot(dates, y_true, label="真实值", lw=1.2, color="#333")
    axes[0].plot(dates, preds["linear"], label=f"线性 RMSE={results['linear']['RMSE']}", lw=1.0, alpha=0.85)
    axes[0].plot(dates, preds["mlp"], label=f"MLP RMSE={results['mlp']['RMSE']}", lw=1.0, alpha=0.85)
    axes[0].set_title("主测试集到件量预测（时间序列切分，目标未缩尾）")
    axes[0].tick_params(axis="x", rotation=25, labelsize=7)
    axes[0].legend(fontsize=8, frameon=False)

    resid = y_true.to_numpy() - preds[best]
    axes[1].scatter(preds[best], resid, s=12, alpha=0.7, color="#4C78A8")
    axes[1].axhline(0, color="red", lw=0.8, ls="--")
    axes[1].set_title(f"残差图（{results[best]['name']}）")
    axes[1].set_xlabel("预测值")
    axes[1].set_ylabel("残差")
    fig.tight_layout()
    fig.savefig(FIGURES / "regression.png")
    plt.close(fig)

    # ---------------- 最差样本与校历切换窗口
    test = df.iloc[split:].copy().reset_index(drop=True)
    test["pred"] = preds[best]
    test["abs_err"] = (test["volume"] - test["pred"]).abs()
    worst = test.nlargest(3, "abs_err")[["date", "volume", "pred", "abs_err", "promo_index", "is_vacation"]]
    worst_records = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "volume": int(r["volume"]),
            "pred": round(float(r["pred"]), 1),
            "abs_err": round(float(r["abs_err"]), 1),
            "promo_index": float(r["promo_index"]),   # 保留原值，不随误差列一起四舍五入
            "is_vacation": int(r["is_vacation"]),
        }
        for _, r in worst.iterrows()
    ]

    # 切换窗口：按方向拆分、窗口 14 天、报有符号偏差（正数=低估，负数=高估）
    pred_full = best_pipe.predict(design(df))
    df["pred_full"] = pred_full
    df["signed_err"] = df["volume"] - df["pred_full"]
    vac_all = df["is_vacation"].to_numpy()
    switch_idx = np.where(np.diff(vac_all) != 0)[0] + 1
    windows, outside_mask = [], np.zeros(len(df), dtype=bool)
    for i in switch_idx:
        end = min(i + 14, len(df))
        direction = "开学（假期→学期）" if vac_all[i] == 0 else "放假（学期→假期）"
        mask = np.zeros(len(df), dtype=bool)
        mask[i:end] = True
        outside_mask |= mask
        m_test = mask & (np.arange(len(df)) >= split)
        if m_test.sum() == 0:
            continue
        windows.append({
            "切换日": df["date"].iloc[i].strftime("%Y-%m-%d"),
            "方向": direction,
            "窗口": f"{df['date'].iloc[i]:%m-%d} ~ {df['date'].iloc[end-1]:%m-%d}",
            "窗口天数": int(mask.sum()),
            "落在测试集天数": int(m_test.sum()),
            "有符号偏差": round(float(df.loc[m_test, "signed_err"].mean()), 1),
            "MAE": round(float(df.loc[m_test, "signed_err"].abs().mean()), 1),
        })
    test_mask = np.arange(len(df)) >= split
    normal = test_mask & ~outside_mask
    excluded = df.loc[test_mask & outside_mask]
    cal = {
        "switches": windows,
        "窗口外（测试集）": {
            "天数": int(normal.sum()),
            "有符号偏差": round(float(df.loc[normal, "signed_err"].mean()), 2),
            "MAE": round(float(df.loc[normal, "signed_err"].abs().mean()), 1),
        },
        "剔除切换窗口后的测试集": {
            "天数": int(normal.sum()),
            "RMSE": round(float(np.sqrt((df.loc[normal, "signed_err"] ** 2).mean())), 2),
            "MAE": round(float(df.loc[normal, "signed_err"].abs().mean()), 1),
        },
        "口径说明": "有符号偏差 = 真实 − 预测，**正数表示低估**（真实高于预测），负数表示高估。"
                    "窗口为切换日起 14 天，按入假/开学两个方向分别统计，避免把两次方向相反的切换混成一个数。",
    }

    # lag1 系数在"有无校历特征"两种设定下的对比（遗漏变量偏差的直接证据）
    def _lag1_coef(use_vac: bool) -> float:
        Xc = pd.concat([
            df[["is_weekend", "is_holiday", "promo_index", "temp_c", "rain_mm",
                "lag1_volume", "lag7_volume", "roll7_mean", "month"]],
            pd.get_dummies(df["dow"], prefix="dow", dtype=float),
        ], axis=1)
        if use_vac:
            Xc["is_vacation"] = df["is_vacation"]
        pipe = Pipeline([("s", StandardScaler()), ("l", LinearRegression())]).fit(Xc, df["volume"])
        return round(float(pipe.named_steps["l"].coef_[list(Xc.columns).index("lag1_volume")]), 1)

    lag_coef = {"without_vacation": _lag1_coef(False), "with_vacation": _lag1_coef(True)}

    # 开学窗口的前段（尚无促销）单独统计，用于区分"校历切换"与"促销爬坡"
    open_win = next((w for w in windows if "开学" in w["方向"]), None)
    if open_win:
        d0 = pd.Timestamp("2026-08-26")
        pre = (df["date"] >= d0) & (df["date"] <= d0 + pd.Timedelta(days=5))
        open_win["前段（无促销）"] = {
            "区间": "08-26 ~ 08-31",
            "天数": int(pre.sum()),
            "有符号偏差": round(float(df.loc[pre, "signed_err"].mean()), 1),
        }

    out = {
        "rows": len(df), "train": split, "test": len(df) - split,
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "feature_set": FEATURE_ORDER,
        "models": results, "best": best,
        "ablation_rmse": ablation,
        "same_level_peak_618": same_level,
        "extrapolation_1111": extrapolation,
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_corr_with_volume": corr,
        "pca_loadings": loadings.to_dict(),
        "collinearity_fix": collinearity,
        "lag1_coef": lag_coef,
        "worst_cases": worst_records,
        "calendar_switch_windows": cal,
    }
    save_json(out, METRICS / "module3.json")

    print(json.dumps({"models": results, "best": best, "ablation": ablation,
                      "same_level_618": same_level["linear"],
                      "month_encoding_618": same_level["month_encoding"],
                      "ladder": {k: (v["RMSE"], v["R2"]) for k, v in ladder.items()},
                      "calendar": cal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
