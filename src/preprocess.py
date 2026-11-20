"""数据清洗与预处理：输出干净数据集、清洗前后统计、数据概览图。

对应评分项「数据方案与预处理」(15%)：每一步清洗都记录前后数量，
生成《数据方案与清洗报告》所需的全部数字。
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import (
    DATA_PROC,
    DATA_RAW,
    FIGURES,
    TEXT_MIN_LEN,
    ensure_dirs,
    normalize_text,
    save_json,
    setup_matplotlib,
)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ALNUM = re.compile(r"[0-9A-Za-z]")


def _is_garbage(text: str, min_cn: int = 2, max_junk_ratio: float = 0.35, alnum_exempt: int = 4) -> bool:
    """乱码判定：以"是否含足够中文"为主，符号占比为辅。

    旧规则用"中文占比 < 50%"，会把含运单号/取件码的**正常工单**误杀：
    `取件码123456`（中文占比 33%）、`SF1234567890到哪了`（20%）都会被删。
    新规则改为绝对数量判定：
      1) 中文字符 < 2 → 无可用语义，剔除（asdf / 。。。 / □□□□ / 纯数字）；
      2) 否则仅在"符号占比高且字母数字很少"时剔除（取件码%#@ / 。。abc）；
         字母数字 ≥ 4 的一律放行，保护运单号查询类工单。
    """
    n = len(text)
    if n == 0:
        return True
    cn = len(_CJK.findall(text))
    alnum = len(_ALNUM.findall(text))
    if cn < min_cn:
        return True
    return (1 - (cn + alnum) / n) > max_junk_ratio and alnum < alnum_exempt


# ------------------------------------------------------------------ 工单
def clean_tickets() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_RAW / "tickets.csv", encoding="utf-8-sig")
    n0 = len(raw)
    stats: dict = {"raw_rows": n0}

    # 1) 缺失与空值：直接剔除，后续统计可逐项相加
    raw["text"] = raw["text"].fillna("").astype(str)
    raw["label"] = raw["label"].fillna("other")
    n_missing = int((raw["text"].str.strip() == "").sum())
    raw = raw[raw["text"].str.strip() != ""].reset_index(drop=True)

    # 2) 文本归一化（全半角、表情、空白、常见口语错字）
    raw["text_clean"] = raw["text"].map(normalize_text)

    # 3) 去重（同一条工单重复上报，取首次出现）
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["text_clean"], keep="first").reset_index(drop=True)
    n_dup = n_before - len(raw)

    # 4) 过短文本（"?" "1" 这类无法判类的记录；阈值保持 2，"查件""退款" 合法保留）
    n_short = int((raw["text_clean"].str.len() < TEXT_MIN_LEN).sum())
    raw = raw[raw["text_clean"].str.len() >= TEXT_MIN_LEN].reset_index(drop=True)

    # 5) 乱码剔除（中文占比过低 / 符号占比过高），不靠提高长度阈值
    garbage_mask = raw["text_clean"].map(_is_garbage)
    n_garbage = int(garbage_mask.sum())
    garbage_rows = raw.loc[garbage_mask, "text_clean"].tolist()
    raw = raw[~garbage_mask].reset_index(drop=True)

    # 6) 超长文本打标（不删除，交给推理阶段截断）
    raw["is_long"] = (raw["text_clean"].str.len() > 200).astype(int)
    raw["text_len"] = raw["text_clean"].str.len()

    stats.update(
        {
            "missing_text": n_missing,
            "duplicates_removed": n_dup,
            "too_short_removed": n_short,
            "garbage_removed": n_garbage,
            "garbage_examples": garbage_rows[:6],
            "clean_rows": len(raw),
            "label_dist": raw["label_name"].value_counts().to_dict(),
            "text_len": {
                "mean": round(float(raw["text_len"].mean()), 2),
                "p50": int(raw["text_len"].median()),
                "p95": int(raw["text_len"].quantile(0.95)),
                "max": int(raw["text_len"].max()),
            },
            "long_text_rows": int(raw["is_long"].sum()),
        }
    )
    return raw, stats


# ------------------------------------------------------------------ 用户
def clean_users() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_RAW / "users.csv", encoding="utf-8-sig")
    stats: dict = {"raw_rows": len(df)}
    feat = [
        "weekly_pickups", "avg_dwell_hours", "night_ratio", "weekend_ratio",
        "proxy_ratio", "large_item_ratio", "complaint_cnt", "avg_weight_kg",
    ]
    # 比例类特征裁剪到 [0,1]，避免负值/超界
    for c in ["night_ratio", "weekend_ratio", "proxy_ratio", "large_item_ratio"]:
        df[c] = df[c].clip(0, 1)
    df["weekly_pickups"] = df["weekly_pickups"].clip(0, 12)
    df["avg_dwell_hours"] = df["avg_dwell_hours"].clip(0.5, 120)
    df["avg_weight_kg"] = df["avg_weight_kg"].clip(0.1, 8)
    df["complaint_cnt"] = df["complaint_cnt"].clip(0, 8)

    # 3σ 规则检测异常值（只记录，不删除：聚类对极端值不敏感，且异常值本身是业务信号）
    outlier = {}
    for c in feat:
        mu, sd = df[c].mean(), df[c].std()
        outlier[c] = int(((df[c] - mu).abs() > 3 * sd).sum())
    stats.update({"outliers_3sigma": outlier, "clean_rows": len(df)})
    return df, stats


# ------------------------------------------------------------------ 到件量
def clean_volume() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_RAW / "daily_volume.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    stats: dict = {"raw_rows": len(df)}

    # 缺失日期补齐（真实业务中系统宕机会造成缺日）
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_days = len(full) - df["date"].nunique()
    df = df.set_index("date").reindex(full).rename_axis("date").reset_index()
    df["volume"] = df["volume"].interpolate(limit_direction="both")

    # 补齐日的日历字段必须按日期重算，不能用邻日 bfill/ffill（否则星期会错位）
    from generate_data import HOLIDAYS, _promo_index, _vacation_factor  # 与生成器共用同一套日历规则

    df["dow"] = df["date"].dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["is_holiday"] = df.apply(lambda r: int((r["date"].month, r["date"].day) in HOLIDAYS), axis=1)
    df["promo_index"] = df["date"].map(lambda d: _promo_index(d.to_pydatetime()))
    df["is_vacation"] = df["date"].map(lambda d: int(_vacation_factor(d.to_pydatetime()) < 1.0))
    # 气温/降雨属观测值，缺失日用邻日插值是可接受的
    df["temp_c"] = df["temp_c"].interpolate(limit_direction="both")
    df["rain_mm"] = df["rain_mm"].fillna(0.0)

    # IQR 只用于标记离群点，不再缩尾：双 11 峰值是真实信号，不是异常值
    q1, q3 = df["volume"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_outlier"] = ((df["volume"] < lo) | (df["volume"] > hi)).astype(int)
    n_out = int(df["is_outlier"].sum())
    n_out_promo = int(df.loc[df["is_outlier"] == 1, "promo_index"].gt(0).sum())
    df["volume_raw"] = df["volume"]

    # 重建滞后特征（补齐日期后原滞后列已失效）
    df["lag1_volume"] = df["volume"].shift(1)
    df["lag7_volume"] = df["volume"].shift(7)
    df["roll7_mean"] = df["volume"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)

    stats.update(
        {
            "missing_days_filled": int(missing_days),
            "outliers_flagged": n_out,
            "outliers_on_promo_days": n_out_promo,
            "outliers_winsorized": 0,
            "clean_rows": len(df),
            "volume_mean": round(float(df["volume"].mean()), 1),
            "volume_max": int(df["volume"].max()),
            "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        }
    )
    return df, stats


# ------------------------------------------------------------------ 概览图
def plot_overview(tickets: pd.DataFrame, volume: pd.DataFrame) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    vc = tickets["label_name"].value_counts()
    axes[0].barh(vc.index[::-1], vc.values[::-1], color="#4C78A8")
    axes[0].set_title("工单类别分布（清洗后）")
    axes[0].set_xlabel("条数")

    axes[1].hist(tickets["text_len"], bins=20, color="#F58518", edgecolor="white")
    axes[1].set_title("工单文本长度分布")
    axes[1].set_xlabel("字符数")
    axes[1].set_ylabel("频数")

    axes[2].plot(volume["date"], volume["volume"], lw=0.9, color="#54A24B")
    axes[2].set_title("日到件量时序（含双 11 峰值）")
    axes[2].tick_params(axis="x", rotation=30, labelsize=7)
    axes[2].set_ylabel("件/天")

    fig.tight_layout()
    fig.savefig(FIGURES / "data_overview.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    tickets, s1 = clean_tickets()
    users, s2 = clean_users()
    volume, s3 = clean_volume()

    tickets.to_csv(DATA_PROC / "tickets_clean.csv", index=False, encoding="utf-8-sig")
    users.to_csv(DATA_PROC / "users_clean.csv", index=False, encoding="utf-8-sig")
    volume.to_csv(DATA_PROC / "volume_clean.csv", index=False, encoding="utf-8-sig")

    stats = {"tickets": s1, "users": s2, "volume": s3}
    save_json(stats, DATA_PROC / "cleaning_stats.json")
    plot_overview(tickets, volume)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n[ok] 清洗后数据 -> data/processed/，概览图 -> outputs/figures/data_overview.png")


if __name__ == "__main__":
    main()
