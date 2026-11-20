"""模块四：驿站政策智能问答（大模型 + 检索增强 + 三种提示策略）

设计要点
--------
1. 先检索后生成：用 TF-IDF 从 20 条政策知识库中检索 Top-2 条款作为上下文，
   降低大模型编造政策的概率；
2. 三种提示策略：零样本 / 少样本 / 思维链，除示例与推理指令外其余条件完全一致；
3. 离线降级：未配置 API Key 或调用失败时，直接返回知识库检索答案并标记 mode，
   保证系统在答辩现场断网也能演示（对应「异常处理与鲁棒性设计」）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import KNOWLEDGE, METRICS, ROOT, ensure_dirs, load_json, save_json

try:  # 可选：从项目根目录 .env 读取 Key
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

STRATEGIES = {
    "zero": "零样本",
    "few": "少样本",
    "cot": "思维链",
}

SYSTEM_PROMPT = (
    "你是「校园快递驿站」的智能客服。回答必须准确、简洁、可执行，"
    "只依据给定的参考资料作答；资料未覆盖时明确说明并建议联系人工客服，不得编造政策。"
)

FEW_SHOT = [
    ("包裹在驿站最多能放几天？", "免费保管 3 天，第 4 天起收取超时保管费 1 元/天，单件最高 5 元；超过 7 天未取将退回发件方。"),
    ("我想让室友帮我取件，需要什么手续？", "需提供取件码，并在小程序内将包裹「授权代取」给指定手机号，代取人凭授权码取件。"),
    ("寄到外省首重多少钱？", "跨省首重 12 元/公斤，续重 5 元/公斤，校园网点寄件享 9 折。"),
]


class KnowledgeBase:
    """轻量检索器：字符 n-gram TF-IDF + 余弦相似度。"""

    def __init__(self, faq_path: Path | None = None):
        self.faq = load_json(faq_path or KNOWLEDGE / "faq.json")
        docs = [
            f"{f['title']}。问：{'；'.join(f['questions'])}。答：{f['answer']}" for f in self.faq
        ]
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
        self.mat = self.vec.fit_transform(docs)

    def search(self, question: str, top_k: int = 2) -> list[dict]:
        q = self.vec.transform([question])
        sim = cosine_similarity(q, self.mat)[0]
        idx = np.argsort(-sim)[:top_k]
        return [
            {**self.faq[i], "score": round(float(sim[i]), 3)}
            for i in idx
            if sim[i] > 0.02
        ]

    def context(self, question: str, top_k: int = 2) -> tuple[str, list[dict]]:
        hits = self.search(question, top_k)
        if not hits:
            return "（无匹配条款）", []
        ctx = "\n".join(f"[{h['id']}·{h['title']}] {h['answer']}" for h in hits)
        return ctx, hits


class QAEngine:
    def __init__(self, cache_path: Path | None = None):
        ensure_dirs()
        self.kb = KnowledgeBase()
        self.cache_path = cache_path or (METRICS / "qa_cache.json")
        self.cache: dict = self._load_cache()
        self.api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
            or ""
        )
        # 端点与模型按"实际用的哪家 Key"给默认值；显式配置的 LLM_BASE_URL / LLM_MODEL 优先
        if os.getenv("LLM_BASE_URL"):
            self.base_url = os.getenv("LLM_BASE_URL")
        elif os.getenv("DASHSCOPE_API_KEY"):
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif os.getenv("OPENAI_API_KEY"):
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = "https://open.bigmodel.cn/api/paas/v4"

        if os.getenv("LLM_MODEL"):
            self.model = os.getenv("LLM_MODEL")
        elif os.getenv("DASHSCOPE_API_KEY"):
            self.model = "qwen-plus"
        elif os.getenv("OPENAI_API_KEY"):
            self.model = "gpt-4o-mini"
        else:
            self.model = "glm-4-flash"
        # 深度思考模型（如 qwen4-max）需要 enable_thinking + 流式才能拿到 reasoning_content；
        # 默认关闭：三策略对比比的是"提示词"，不是"模型是否开思考"，混在一起会污染结论。
        self.enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "800"))
        self.timeout = float(os.getenv("LLM_TIMEOUT", "60"))
        self._client = None

    # ------------------------------------------------------------ 缓存
    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        save_json(self.cache, self.cache_path)

    # ------------------------------------------------------------ 提示构造
    def build_messages(self, question: str, strategy: str) -> list[dict]:
        ctx, _ = self.kb.context(question)
        if strategy == "zero":
            user = f"参考资料：\n{ctx}\n\n用户问题：{question}\n\n请直接回答，不超过 120 字。"
        elif strategy == "few":
            examples = "\n\n".join(f"问：{q}\n答：{a}" for q, a in FEW_SHOT)
            user = (
                f"参考资料：\n{ctx}\n\n以下是回答示例：\n{examples}\n\n"
                f"用户问题：{question}\n\n请参照示例的风格直接回答，不超过 120 字。"
            )
        elif strategy == "cot":
            user = (
                f"参考资料：\n{ctx}\n\n用户问题：{question}\n\n"
                "请按以下格式回答：\n"
                "推理要点：\n"
                "- 业务类型：（取件/寄件/退换货/赔付/其他）\n"
                "- 适用条款：（引用条款编号与关键数字）\n"
                "- 结论：（一句话）\n"
                "答复：（面向用户，1-2 条可执行建议，不超过 120 字）"
            )
        else:
            raise ValueError(strategy)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    # ------------------------------------------------------------ 调用
    def _call_llm(self, messages: list[dict], timeout: float | None = None) -> tuple[str, str]:
        """调用兼容 OpenAI 协议的端点，返回 (答复, 模型自带的推理内容)。

        流式解析的原因：深度思考模型（qwen4-max 等）的 `reasoning_content`
        只在流式返回里给出，非流式拿不到；同时流式也便于统计首字延迟。
        """
        from openai import OpenAI

        timeout = timeout or self.timeout
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=timeout, max_retries=1)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": True,
            # 必须显式传 false：该端点默认开启思考，不传就一直返回 reasoning_content，
            # 让"关闭思考"的对照实验名不副实（实测：不传 487 字推理，false 为 0 字）。
            "extra_body": {"enable_thinking": bool(self.enable_thinking)},
        }

        stream = self._client.chat.completions.create(**kwargs)
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
            content = getattr(delta, "content", None)
            if content:
                answer_parts.append(content)
        return "".join(answer_parts).strip(), "".join(reasoning_parts).strip()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"^答复[:：]\s*", "", text.strip())
        return re.sub(r"\n{2,}", "\n", text).strip()

    @staticmethod
    def _split_cot(text: str) -> tuple[str, str]:
        """把思维链输出拆成（推理要点, 面向用户答复）。

        显式推理是 CoT 与零样本的核心差异：若要求模型"不要展示推理过程"，
        对比就退化成"指令详细程度 vs 零样本"，构念效度会被质疑。
        """
        parts = re.split(r"答复[:：]", text, maxsplit=1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return "", text.strip()

    def ask(self, question: str, strategy: str = "cot", use_cache: bool = True) -> dict:
        question = (question or "").strip()
        if not question:
            return {"answer": "请输入问题，例如：包裹最多能放几天？", "mode": "input-error",
                    "strategy": strategy, "latency_ms": 0.0, "sources": []}
        if len(question) > 200:
            question = question[:200]

        ctx, hits = self.kb.context(question)
        key = hashlib.sha1(f"{strategy}|{self.model}|{question}|{ctx}".encode()).hexdigest()

        if use_cache and key in self.cache:
            c = self.cache[key]
            c.update({"cached": True, "sources": [h["id"] for h in hits]})
            return c

        t0 = time.perf_counter()
        if not self.api_key:
            # 离线降级：直接用检索到的条款作答，保证可用性
            ans = hits[0]["answer"] if hits else "抱歉，没有检索到相关政策，建议联系驿站人工客服（8:00–22:00）。"
            out = {
                "answer": f"【离线降级模式·知识库检索】{ans}",
                "reasoning": "",
                "mode": "offline-fallback",
                "strategy": strategy,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "sources": [h["id"] for h in hits],
                "cached": False,
            }
            return out

        try:
            raw_answer, raw_reasoning = self._call_llm(self.build_messages(question, strategy))
            if strategy == "cot":
                split_reason, split_answer = self._split_cot(raw_answer)
                answer = split_answer or self._clean(raw_answer)
                reasoning = raw_reasoning or split_reason
            else:
                answer, reasoning = self._clean(raw_answer), raw_reasoning
            out = {
                "answer": answer,
                "reasoning": reasoning,
                "mode": "llm",
                "strategy": strategy,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "sources": [h["id"] for h in hits],
                "cached": False,
            }
        except Exception as exc:  # 网络/额度/超时统一降级
            ans = hits[0]["answer"] if hits else "抱歉，暂时无法回答，请稍后再试或联系人工客服。"
            out = {
                "answer": f"【降级模式·{type(exc).__name__}】{ans}",
                "reasoning": "",
                "mode": "api-failed",
                "strategy": strategy,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "sources": [h["id"] for h in hits],
                "error": str(exc)[:200],
                "cached": False,
            }
        self.cache[key] = out
        self._save_cache()
        return out


if __name__ == "__main__":
    eng = QAEngine()
    for s in ("zero", "few", "cot"):
        r = eng.ask("包裹在驿站最多能放几天？超时怎么收费？", strategy=s)
        print(f"[{STRATEGIES[s]}] mode={r['mode']} {r['latency_ms']}ms\n{r['answer']}\n")
