"""模块四评测：三种提示策略在 15 个课程/业务问题上的对比

自动指标：关键词覆盖率（参考答案关键信息点是否命中）、平均响应时间、降级率
人工指标：准确性/完整性/可读性（1–5 分），输出 CSV 模板由人工填写

说明：未配置 API Key 时脚本以「降级模式」运行，仅验证链路可用性，
结果不能作为大模型效果结论。
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from common import METRICS, ensure_dirs, save_json
from module4_qa import QAEngine, STRATEGIES

# (问题, 参考答案关键点, 业务类别)
TEST_SET = [
    ("我的包裹最多能在驿站放几天？", ["3 天", "1 元", "5 元", "7 天"], "取件"),
    ("取件码一直没收到怎么办？", ["短信", "小程序", "手机号", "柜台"], "取件"),
    ("可以让同学帮我取快递吗？", ["授权代取", "取件码", "手机号"], "取件"),
    ("驿站晚上几点关门？", ["22:00", "8:00"], "其他"),
    ("寄到外省怎么收费？", ["12 元", "5 元", "9 折"], "寄件"),
    ("充电宝可以寄吗？", ["100Wh", "3C", "陆运", "禁寄"], "寄件"),
    ("保价费怎么算？", ["0.5%", "1 元"], "寄件"),
    ("包裹丢了怎么赔？", ["48 小时", "工单", "3 个工作日"], "赔付"),
    ("没保价的件坏了赔多少？", ["7 倍", "运费"], "赔付"),
    ("退货怎么寄回？", ["退货单号", "二维码", "扫码"], "退换"),
    ("想改收货地址可以吗？", ["未揽收", "转寄费"], "寄件"),
    ("发票怎么开？", ["3 个工作日", "电子普票"], "其他"),
    ("怎么加盟驿站？", ["20", "300"], "其他"),
    ("会员有什么优惠？", ["9.9", "3 次", "8.5 折"], "其他"),
    ("包裹被放进错误的柜子怎么办？", ["柜机编号", "30 分钟"], "取件"),
]


def coverage(answer: str, keywords: list[str]) -> float:
    if not answer:
        return 0.0
    hit = sum(1 for k in keywords if k.replace(" ", "") in answer.replace(" ", ""))
    return round(hit / len(keywords), 3)


def main(argv: list[str] | None = None) -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser(description="三种提示策略对比评测")
    ap.add_argument("--require-llm", action="store_true",
                    help="未配置 API Key 时直接退出（用于提交前确认模块四真正跑通）")
    ap.add_argument("--fresh", action="store_true", help="清空响应缓存后重新调用")
    args = ap.parse_args(argv)

    eng = QAEngine()
    if args.require_llm and not eng.api_key:
        print("[FATAL] 未检测到 API Key：模块四无法产生真实的三策略对比。\n"
              "        请先设置 DASHSCOPE_API_KEY 或 ZHIPUAI_API_KEY，再运行本脚本。")
        raise SystemExit(2)
    if args.fresh and eng.cache_path.exists():
        eng.cache_path.unlink()
        eng.cache = {}

    rows = []
    for qid, (q, kws, cat) in enumerate(TEST_SET, 1):
        for s, cn in STRATEGIES.items():
            r = eng.ask(q, strategy=s)
            rows.append(
                {
                    "qid": qid,
                    "question": q,
                    "category": cat,
                    "strategy": cn,
                    "answer": r["answer"],
                    "reasoning": r.get("reasoning", "")[:300],
                    "mode": r["mode"],
                    "latency_ms": r["latency_ms"],
                    "sources": ",".join(r.get("sources", [])),
                    "keyword_coverage": coverage(r["answer"], kws),
                    "score_accuracy_1_5": "",
                    "score_completeness_1_5": "",
                    "score_readability_1_5": "",
                }
            )

    df = pd.DataFrame(rows)
    valid = bool((df["mode"] == "llm").any())
    new_llm = int((df["mode"] == "llm").sum())

    # 覆盖护栏：一次"无 Key / 超时 / 额度耗尽"的降级运行，不能把已有的真实对比结果冲掉。
    # 判据不是"有没有 Key"，而是"本次是否真的产生了 LLM 调用"。
    old = {}
    if (METRICS / "module4.json").exists():
        try:
            old = json.loads((METRICS / "module4.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
    old_llm = int(old.get("llm_calls", 0) or 0)
    degraded = new_llm == 0 and old_llm > 0

    # 格式类派生列：关键词覆盖率会饱和（三策略同为 0.9），差异主要体现在格式与延迟上
    df["answer_len"] = df["answer"].str.len()
    df["prefix_leak"] = df["answer"].str.startswith("答：").astype(int)      # 少样本把示例前缀学了过去
    df["numbered"] = df["answer"].str.match(r"^\s*\d+[.、]").astype(int)    # 思维链的编号步骤格式

    if degraded:
        df.to_csv(METRICS / "qa_eval.degraded.csv", index=False, encoding="utf-8-sig")
        summary_path = METRICS / "module4.degraded.json"
        print(f"[护栏] 本次 0 次 LLM 调用，而已有结果含 {old_llm} 次真实调用；"
              f"降级结果另存为 {summary_path.name}，不覆盖 module4.json")
    else:
        df.to_csv(METRICS / "qa_eval.csv", index=False, encoding="utf-8-sig")
        summary_path = METRICS / "module4.json"

    _bs = (
        df.groupby("strategy")
        .agg(
            avg_keyword_coverage=("keyword_coverage", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
            avg_answer_len=("answer_len", "mean"),
            prefix_leak_n=("prefix_leak", "sum"),
            numbered_n=("numbered", "sum"),
        )
        .round(3)
        .to_dict(orient="index")
    )
    ratio = None
    if "思维链" in _bs and "零样本" in _bs:
        ratio = round(_bs["思维链"]["avg_latency_ms"] / _bs["零样本"]["avg_latency_ms"], 2)

    summary = {
        "n_questions": len(TEST_SET),
        "n_calls": len(df),
        "llm_calls": new_llm,
        "fallback_calls": int((df["mode"] != "llm").sum()),
        "valid_for_conclusion": valid,
        "model": eng.model,
        "by_strategy": _bs,
        "cot_vs_zero_latency_ratio": ratio,
    }
    save_json(summary, summary_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        print("\n[注意] 未配置 API Key，本次为离线降级链路自检，关键词覆盖率不作为模型效果结论。")
        print("      配置 DASHSCOPE_API_KEY 或 ZHIPUAI_API_KEY 后重新运行本脚本即可得到真实对比。")
    print(f"\n评测明细（含人工打分空列）已写入 {METRICS / 'qa_eval.csv'}")


if __name__ == "__main__":
    main()
