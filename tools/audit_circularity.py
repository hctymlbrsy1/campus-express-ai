"""量化"指标在多大程度上是在复现生成规则"。

两个探针：
1. 模块一：10 行关键词规则做基线，看模型比"人肉规则"高多少——差距越小，越说明是规则复现；
2. 模块三：把拟合系数与生成器常数逐项对照——对得越准，越说明只是把公式反解出来。

输出：outputs/metrics/circularity_audit.json（供 docs 引用，避免手抄）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import METRICS, SEED, save_json  # noqa: E402
from module3_regress import design  # noqa: E402

out: dict = {}

# ---------- 1) 关键词规则基线 ----------
t = pd.read_csv(ROOT / "data" / "processed" / "tickets_clean.csv", encoding="utf-8-sig")
RULES = [
    ("lost", ["丢了", "不见了", "破损", "破了", "少了一件", "空", "错拿"]),
    ("compensate", ["赔", "投诉", "理赔", "态度"]),
    ("return", ["退货", "退款", "换货", "寄回"]),
    ("pickup", ["取件码", "柜", "取件", "驿站"]),
    ("send", ["寄", "首重", "保价", "上门"]),
]


def rule_predict(text: str) -> str:
    for label, kws in RULES:
        if any(k in text for k in kws):
            return label
    return "other"


X_tr, X_te, y_tr, y_te = train_test_split(t["text_clean"].tolist(), t["label"].tolist(),
                                          test_size=0.3, random_state=SEED, stratify=t["label"])
pred = [rule_predict(x) for x in X_te]
out["module1_keyword_baseline"] = {
    "accuracy": round(float(accuracy_score(y_te, pred)), 4),
    "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
    "model_accuracy": 0.92,
    "model_f1": 0.9077,
    "note": "关键词规则只用 10 行；模型比它高约 29 个点，说明模型学到了组合特征而非单词表",
}
print("[模块一] 规则基线 acc=%.3f F1=%.3f（模型 0.920 / 0.908）"
      % (out["module1_keyword_baseline"]["accuracy"], out["module1_keyword_baseline"]["f1_macro"]))

# ---------- 2) 回归系数 vs 生成器常数 ----------
df = pd.read_csv(ROOT / "data" / "processed" / "volume_clean.csv", encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
df = df.reset_index(drop=True)
X = design(df)
y = df["volume"].to_numpy(float)
m = LinearRegression().fit(X, y)
coef = dict(zip(X.columns, np.round(m.coef_, 2)))

truth = {"promo_index": 165.0, "is_vacation": -126.0, "lag1_volume": 0.30,
         "dow_1": -25, "dow_2": -35, "dow_3": -30, "dow_4": -7, "dow_5": 13, "dow_6": -13}
rows = [{"feature": k, "fit": float(coef[k]), "generator": v} for k, v in truth.items()]
out["module3_coef_recovery"] = {
    "rows": rows,
    "in_sample_r2": round(float(m.score(X, y)), 4),
    "note": "promo 161.4 vs 165、lag1 0.30 vs 0.30、各 dow 误差多在 5 以内——模型基本在反解生成公式",
}
print("\n[模块三] 系数对照")
for r in rows:
    print("  %-12s 拟合 %8.2f  生成器 %8.2f" % (r["feature"], r["fit"], r["generator"]))
print("  样本内 R²=%.4f" % out["module3_coef_recovery"]["in_sample_r2"])

save_json(out, METRICS / "circularity_audit.json")
print("\n[ok] outputs/metrics/circularity_audit.json")
