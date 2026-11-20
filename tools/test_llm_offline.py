"""模块四离线自测：不联网、不花钱，验证流式解析与 CoT 拆分是否正确。

用假客户端模拟两种返回：
  1. 思考模型：先吐 reasoning_content，再吐 content；
  2. 普通模型：只吐 content，CoT 靠提示词里的「答复：」分隔。

用法：python tools/test_llm_offline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["DASHSCOPE_API_KEY"] = "sk-test-dummy"
os.environ.pop("LLM_BASE_URL", None)
os.environ["LLM_ENABLE_THINKING"] = "false"

from module4_qa import QAEngine  # noqa: E402


def chunk(reasoning: str | None = None, content: str | None = None):
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    def create(self, **kwargs):
        assert kwargs.get("stream") is True, "应使用流式调用"
        return iter(self._chunks)


class FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=FakeCompletions(chunks))


def run_case(name: str, strategy: str, chunks, expect_answer: str, expect_reason: str) -> bool:
    eng = QAEngine()
    eng._client = FakeClient(chunks)
    r = eng.ask("包裹最多能放几天？", strategy=strategy, use_cache=False)
    ok = expect_answer in r["answer"] and expect_reason in (r.get("reasoning") or "")
    print(("PASS" if ok else "FAIL"), name, "| mode=", r["mode"], "| answer=", r["answer"][:40],
          "| reasoning=", (r.get("reasoning") or "")[:30])
    return ok


def main() -> int:
    ok = True
    # 1) 零样本：普通模型，只有 content
    ok &= run_case("零样本·普通模型", "zero",
                   [chunk(content="包裹免费保管 3 天。"), chunk(content="超时 1 元/天。")],
                   "包裹免费保管 3 天", "")

    # 2) CoT：模型只给 content，但有「答复：」分隔 → 靠提示词拆分
    ok &= run_case("CoT·无模型推理", "cot",
                   [chunk(content="推理要点：\n- 业务类型：取件\n答复：免费保管 3 天，超时 1 元/天。")],
                   "免费保管 3 天", "业务类型")

    # 3) CoT：思考模型，reasoning_content 与 content 分开 → 优先用模型推理
    ok &= run_case("CoT·思考模型", "cot",
                   [chunk(reasoning="思考：先定业务类型"), chunk(reasoning="，再找条款。"),
                    chunk(content="答复：免费保管 3 天。")],
                   "免费保管 3 天", "先定业务类型")

    print("\n结论：" + ("全部通过" if ok else "存在失败用例"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
