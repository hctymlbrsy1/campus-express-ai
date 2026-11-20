"""模块一：工单文本分类（多分类）

模型：TF-IDF 字符 n-gram + 逻辑回归 / 线性 SVM
产出：指标表、混淆矩阵、错分样本（供失败案例分析）、可复用模型
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common import (
    DATA_PROC,
    FIGURES,
    METRICS,
    MODELS,
    SEED,
    clip_text,
    ensure_dirs,
    load_json,
    normalize_text,
    save_json,
    setup_matplotlib,
    timer,
)

LABEL_ORDER = ["pickup", "send", "return", "lost", "compensate", "other"]
LABEL_CN = {
    "pickup": "取件问题",
    "send": "寄件问题",
    "return": "退换货",
    "lost": "丢件破损",
    "compensate": "赔付投诉",
    "other": "其他咨询",
}

# 关键词规则：既是审计基线（tools/audit_circularity.py），也是模型不可用时的兜底
RULE_KEYWORDS = [
    ("lost", ["丢了", "不见了", "破损", "破了", "少了一件", "空", "错拿"]),
    ("compensate", ["赔", "投诉", "理赔", "态度"]),
    ("return", ["退货", "退款", "换货", "寄回"]),
    ("pickup", ["取件码", "柜", "取件", "驿站"]),
    ("send", ["寄", "首重", "保价", "上门"]),
]


def rule_predict(text: str) -> tuple[str, float]:
    """按关键词命中顺序给一个粗分类；置信度固定 0.5 表示“规则判断，非模型概率”。"""
    for label, kws in RULE_KEYWORDS:
        if any(k in text for k in kws):
            return label, 0.5
    return "other", 0.5


def make_pipeline(kind: str, ngram=(1, 3)) -> Pipeline:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram,
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
    )
    if kind == "lr":
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    elif kind == "svm":
        clf = CalibratedClassifierCV(
            SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=SEED),
            cv=3,
            ensemble=False,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def plot_confusion(cm: np.ndarray, labels_cn: list[str], path, model_name: str = "") -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_cn)), labels_cn, rotation=35, ha="right")
    ax.set_yticks(range(len(labels_cn)), labels_cn)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"工单分类混淆矩阵（{model_name}）")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def skeleton(text: str, n: int) -> str:
    """去数字与标点后的前 n 字骨架，用于近似"同一模板"分组。"""
    t = re.sub(r"\d+", "", normalize_text(text))
    t = re.sub(r"[，。！？、；：\s]", "", t)
    return t[:n]


def group_split_eval(kind: str, X, y, n: int, seeds=(0, 1, 2)) -> dict:
    groups = [skeleton(t, n) for t in X]
    accs, f1s = [], []
    for seed in seeds:
        tr, te = next(
            GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(X, y, groups)
        )
        pipe = make_pipeline(kind).fit([X[i] for i in tr], [y[i] for i in tr])
        pred = pipe.predict([X[i] for i in te])
        accs.append(accuracy_score([y[i] for i in te], pred))
        f1s.append(f1_score([y[i] for i in te], pred, average="macro"))
    return {
        "skeleton_len": n,
        "n_groups": len(set(groups)),
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_std": round(float(np.std(accs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_PROC / "tickets_clean.csv", encoding="utf-8-sig")
    X = df["text_clean"].map(normalize_text).tolist()
    y = df["label"].tolist()

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(df)), test_size=0.3, random_state=SEED, stratify=y
    )

    results, fitted, preds = {}, {}, {}
    for kind, name in [("lr", "逻辑回归"), ("svm", "线性SVM")]:
        pipe = make_pipeline(kind)
        with timer() as t:
            pipe.fit(X_tr, y_tr)
        with timer() as t_inf:
            proba = pipe.predict_proba(X_te)
        pred = pipe.named_steps["clf"].classes_[proba.argmax(1)]
        preds[kind] = (pred, proba)
        fitted[kind] = pipe

        # 5 折交叉验证：检验小样本下的稳定性
        cv = cross_val_score(
            make_pipeline(kind), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="f1_macro", n_jobs=1,
        )
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        results[kind] = {
            "name": name,
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
            "f1_weighted": round(float(f1_score(y_te, pred, average="weighted")), 4),
            "cv5_f1_macro_mean": round(float(cv.mean()), 4),
            "cv5_f1_macro_std": round(float(cv.std()), 4),
            "train_ms": round(t["ms"], 1),
            "infer_ms_per_100": round(t_inf["ms"] / max(len(X_te), 1) * 100, 2),
            "per_class": {
                LABEL_CN[k]: {m: round(v, 3) for m, v in rep[k].items() if m in ("precision", "recall", "f1-score")}
                for k in LABEL_CN
                if k in rep
            },
        }

    # ---------------- 双口径评估：随机切分 vs 模板分组切分
    groups12 = [skeleton(t, 12) for t in X]
    for kind in ("lr", "svm"):
        gk = cross_val_score(
            make_pipeline(kind), X, y, cv=GroupKFold(n_splits=5), groups=groups12,
            scoring="f1_macro", n_jobs=1,
        )
        results[kind]["cv5_group_f1_macro_mean"] = round(float(gk.mean()), 4)
        results[kind]["cv5_group_f1_macro_std"] = round(float(gk.std()), 4)
        results[kind]["group_split"] = group_split_eval(kind, X, y, 12)

    # 骨架长度敏感性：说明"分组切分"本身不是唯一真值
    sensitivity = [group_split_eval("lr", X, y, n) for n in (8, 12, 20)]
    skeleton_range = [min(r["acc_mean"] for r in sensitivity), max(r["acc_mean"] for r in sensitivity)]
    # 跨口径区间：必须同时含随机切分（上界）与骨架分组（下界），
    # 否则字段名 across_protocols 名不副实（曾把 0.920 排除在外）
    all_acc = skeleton_range + [results[k]["accuracy"] for k in results]
    acc_range = [min(all_acc), max(all_acc)]

    # 以 macro-F1 选优
    best = max(results, key=lambda k: results[k]["f1_macro"])
    best_pipe = fitted[best]
    pred, proba = preds[best]
    joblib.dump(best_pipe, MODELS / "ticket_classifier.joblib")
    for k, p in fitted.items():
        joblib.dump(p, MODELS / f"ticket_{k}.joblib")

    labels = LABEL_ORDER
    cm = confusion_matrix(y_te, pred, labels=labels)
    plot_confusion(cm, [LABEL_CN[k] for k in labels], FIGURES / "cm_ticket.png", results[best]["name"])

    # 错分样本：带置信度，供失败案例分析
    conf = proba.max(1)
    err = pd.DataFrame(
        {
            "text": np.array(X_te, dtype=object),
            "true": y_te,
            "pred": pred,
            "confidence": np.round(conf, 3),
            "is_hard": df.iloc[idx_te]["is_hard"].values,
        }
    )
    err = err[err["true"] != err["pred"]].sort_values("confidence", ascending=False)
    err["true_cn"] = err["true"].map(LABEL_CN)
    err["pred_cn"] = err["pred"].map(LABEL_CN)
    err.to_csv(METRICS / "module1_errors.csv", index=False, encoding="utf-8-sig")

    out = {
        "dataset": {"rows": len(df), "train": len(X_tr), "test": len(X_te), "classes": len(set(y))},
        "models": results,
        "best": best,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "error_count": int(len(err)),
        "error_rate": round(float(len(err) / len(X_te)), 4),
        "error_rate_pct": round(float(len(err) / len(X_te)) * 100, 2),
        "hard_sample_error_rate": round(
            float((err["is_hard"] == 1).sum() / max(int((df.iloc[idx_te]["is_hard"] == 1).sum()), 1)), 4
        ),
        "hard_sample_error_rate_pct": round(
            float((err["is_hard"] == 1).sum() / max(int((df.iloc[idx_te]["is_hard"] == 1).sum()), 1)) * 100, 2
        ),
        "evaluation_protocol": {
            "主口径": "随机分层 7:3（同模板变体可能分到两侧，偏乐观）",
            "辅助口径": "按去数字前 12 字骨架分组切分（跨模板，偏保守）",
            "骨架敏感性": sensitivity,
            "acc_range_across_protocols": acc_range,
            "acc_range_across_skeletons": skeleton_range,
            "说明": "两种口径给出的是上下界，真实工单分布介于两者之间；报告中并列呈现，不只报高值。",
        },
    }
    save_json(out, METRICS / "module1.json")
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, ensure_ascii=False, indent=2))
    for k, v in results.items():
        print(f"{v['name']}: acc={v['accuracy']} macroF1={v['f1_macro']} cv5={v['cv5_f1_macro_mean']}±{v['cv5_f1_macro_std']}")
    print(f"最优模型: {results[best]['name']}，错分 {len(err)} 条")


if __name__ == "__main__":
    main()
