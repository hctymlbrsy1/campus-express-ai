"""汇总人工评分（qa_scoring_filled.xlsx）→ outputs/metrics/qa_scores.json。

只读评分表，不改分；缺失行单独列出，均值按已填行计算。
用法：
    python tools/summarize_qa_scores.py
    python tools/summarize_qa_scores.py --input ..\\express_ai_external\\scoring\\qa_scoring_filled.xlsx
评分表默认放在 outputs/metrics/ 下；被移走时用 --input 指过去。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import METRICS, save_json  # noqa: E402

DEFAULT_SRC = METRICS / "qa_scoring_filled.xlsx"
RENAME = {"准确性(1-5)": "accuracy", "完整性(1-5)": "completeness", "可读性(1-5)": "readability"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_SRC), help="评分表 xlsx 路径")
    args = ap.parse_args()
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"找不到评分表：{src}\n若已移出项目，请用 --input 指定路径")
    df = pd.read_excel(src, sheet_name="评分表").rename(columns=RENAME)
    cols = ["accuracy", "completeness", "readability"]

    df["overall"] = df[cols].mean(axis=1)
    filled = df.dropna(subset=cols)
    missing = df[df[cols].isna().any(axis=1)]

    order = ["零样本", "少样本", "思维链"]
    by = {}
    for s in order:
        sub = filled[filled["策略"] == s]
        by[s] = {
            "n": int(len(sub)),
            "accuracy": round(float(sub["accuracy"].mean()), 2),
            "completeness": round(float(sub["completeness"].mean()), 2),
            "readability": round(float(sub["readability"].mean()), 2),
            "overall": round(float(sub["overall"].mean()), 2),
            "std_overall": round(float(sub["overall"].std(ddof=0)), 2),
        }

    out = {
        "source": src.name,
        "scored_rows": int(len(filled)),
        "total_rows": int(len(df)),
        "missing": [
            {"qid": int(r["题号"]), "strategy": r["策略"]} for _, r in missing.iterrows()
        ],
        "by_strategy": by,
        "note": "准确性 42 行全为 5 分；差异集中在完整性与可读性。缺失行按空值计算，不参与均值。",
    }
    save_json(out, METRICS / "qa_scores.json")

    print(f"已填 {len(filled)}/{len(df)} 行，缺失 {len(missing)} 行："
          + ("、".join(f"Q{r['题号']}{r['策略']}" for _, r in missing.iterrows()) or "无"))
    print("\n策略    准确性  完整性  可读性  综合(n)")
    for s, v in by.items():
        print(f"{s:<6} {v['accuracy']:>5.2f}  {v['completeness']:>6.2f} {v['readability']:>6.2f}  "
              f"{v['overall']:>5.2f}({v['n']})")


if __name__ == "__main__":
    main()
