"""模块二：用户行为聚类与群体画像

方法：标准化 + 肘部法/轮廓系数选 K + KMeans + PCA 二维可视化
产出：分群画像表、推荐策略、聚类质量指标（轮廓系数/ARI）
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import DATA_PROC, FIGURES, MODELS, METRICS, SEED, ensure_dirs, save_json, setup_matplotlib

FEATURES = [
    "weekly_pickups", "avg_dwell_hours", "night_ratio", "weekend_ratio",
    "proxy_ratio", "large_item_ratio", "complaint_cnt", "avg_weight_kg",
]
FEATURE_CN = {
    "weekly_pickups": "周取件次数",
    "avg_dwell_hours": "平均滞留时长(h)",
    "night_ratio": "夜间取件占比",
    "weekend_ratio": "周末取件占比",
    "proxy_ratio": "代取占比",
    "large_item_ratio": "大件占比",
    "complaint_cnt": "投诉次数",
    "avg_weight_kg": "平均包裹重量(kg)",
}

# 画像命名规则：哪个特征最突出，就叫什么族
PROTOTYPES = [
    ("night_ratio", "夜间取件族", "延长夜间取件窗口，投放 24h 智能柜，21:00 后推送取件提醒"),
    ("proxy_ratio", "代取依赖族", "推广「授权代取」与室友绑定，减少现场排队与二次跑腿"),
    ("avg_dwell_hours", "低频长滞留族", "到期前 24h 短信 + 小程序双提醒，提供超时费减免券"),
    ("weekend_ratio", "周末集中族", "周末增派人手与临时货架，周五晚推送「错峰取件」提醒"),
]


def choose_k(Xs: np.ndarray, k_range=range(2, 9)) -> tuple[list[dict], int]:
    rows, best_k, best_s = [], 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        rows.append({"k": k, "inertia": round(float(km.inertia_), 2), "silhouette": round(float(s), 4)})
        if s > best_s:
            best_s, best_k = s, k
    return rows, best_k


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """按各簇在关键特征上的 z 分数命名，保证名字不重复。"""
    key = profiles[["night_ratio", "proxy_ratio", "avg_dwell_hours", "weekend_ratio"]].copy()
    z = (key - key.mean()) / key.std(ddof=0).replace(0, 1)
    used, names = set(), {}
    for cid in profiles.index:
        for feat, cn, _ in PROTOTYPES:
            if feat in z.columns and z.loc[cid, feat] > 0.35:
                if cn not in used:
                    names[cid] = cn
                    used.add(cn)
                    break
        else:
            names[cid] = f"均衡型群体{cid}"
    # 兜底去重
    seen: dict[str, int] = {}
    for cid, nm in names.items():
        if names[cid].startswith("均衡型"):
            continue
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[cid] = f"{nm}{seen[nm]}"
    return names


def plot_cluster(k_rows: list[dict], Xs: np.ndarray, labels: np.ndarray, pca: PCA, names: dict, path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ks = [r["k"] for r in k_rows]
    axes[0].plot(ks, [r["inertia"] for r in k_rows], "o-")
    axes[0].set_title("肘部法：簇内平方和")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, [r["silhouette"] for r in k_rows], "s-", color="#F58518")
    axes[1].set_title("轮廓系数")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].axhline(max(r["silhouette"] for r in k_rows), ls="--", lw=0.8, color="gray")

    Z = pca.transform(Xs)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for cid in sorted(set(labels)):
        m = labels == cid
        axes[2].scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.75, color=colors[cid % len(colors)],
                        label=f"C{cid} {names.get(cid, '')}")
    axes[2].set_title(f"PCA 二维投影（解释方差 {pca.explained_variance_ratio_[:2].sum():.1%}）")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PROC / "users_clean.csv", encoding="utf-8-sig")
    X = df[FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    k_rows, best_k = choose_k(Xs)
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(Xs)
    labels = km.labels_
    sil = silhouette_score(Xs, labels)
    ari = adjusted_rand_score(df["true_segment"], labels)

    pca = PCA(n_components=2, random_state=SEED).fit(Xs)
    pca_full = PCA(n_components=len(FEATURES), random_state=SEED).fit(Xs)

    prof = df.assign(cluster=labels).groupby("cluster")[FEATURES].mean().round(2)
    prof["人数"] = df.assign(cluster=labels).groupby("cluster").size()
    prof["占比"] = (prof["人数"] / len(df) * 100).round(1)
    names = name_clusters(prof)
    prof.insert(0, "群体名称", [names[i] for i in prof.index])
    prof = prof.rename(columns=FEATURE_CN)

    plot_cluster(k_rows, Xs, labels, pca, names, FIGURES / "cluster_pca.png")

    strategies = {
        names[cid]: next((s for f, cn, s in PROTOTYPES if cn == names[cid]), "维持常规服务，观察后续变化")
        for cid in prof.index
    }

    # PCA 载荷：解释各主成分由哪些行为构成
    loadings = pd.DataFrame(
        pca_full.components_[:3].T, index=[FEATURE_CN[f] for f in FEATURES], columns=["PC1", "PC2", "PC3"]
    ).round(3)

    out = {
        "n_users": len(df),
        "features": FEATURE_CN,
        "k_selection": k_rows,
        "best_k": best_k,
        "silhouette": round(float(sil), 4),
        "ari_vs_true_segment": round(float(ari), 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca_full.explained_variance_ratio_],
        "pca_loadings": loadings.to_dict(),
        "profiles": prof.reset_index().to_dict(orient="records"),
        "strategies": strategies,
        "fit_ms": None,
    }
    save_json(out, METRICS / "module2.json")
    joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "names": names, "features": FEATURES},
                MODELS / "user_segmenter.joblib")
    print(prof.to_string())
    print(f"\nK={best_k}, silhouette={sil:.3f}, ARI(vs 真实分群)={ari:.3f}")
    print("推荐策略:", json.dumps(strategies, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
