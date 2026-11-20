"""公共配置与工具函数。

所有路径均相对于项目根目录 express_ai/，不依赖当前工作目录。
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
KNOWLEDGE = ROOT / "data" / "knowledge"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"

SEED = 42
TEXT_MAX_LEN = 512          # 超长文本截断阈值（字符）
TEXT_MIN_LEN = 2            # 有效文本最短长度


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROC, KNOWLEDGE, MODELS, OUTPUTS, FIGURES, METRICS, ROOT / "logs"):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 文本归一化
_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U0001F1E6-\U0001F1FF" "]+"
)
_WS_RE = re.compile(r"\s+")
# 常见客服口语/错别字归一（只处理高频且无歧义的情况）
_TYPO_MAP = {
    "在吗": "你好",
    "肿么": "怎么",
    "咋": "怎么",
    "神马": "什么",
}


def normalize_text(text: str, *, drop_emoji: bool = True) -> str:
    """文本归一化：全半角统一、去控制符、压缩空白、去表情。

    注意：此处刻意不做分词，分词在特征工程阶段交给字符 n-gram 完成，
    这样对中文错别字和网络用语更鲁棒。
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    if drop_emoji:
        s = _EMOJI_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    for k, v in _TYPO_MAP.items():
        s = s.replace(k, v)
    return s


def clip_text(text: str, max_len: int = TEXT_MAX_LEN) -> tuple[str, bool]:
    """截断超长文本，返回 (文本, 是否被截断)。"""
    if len(text) <= max_len:
        return text, False
    return text[:max_len], True


# ---------------------------------------------------------------- IO
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def timer():
    """用法: with timer() as t: ...; t['ms']"""
    box = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------- 绘图
def setup_matplotlib():
    """配置中文字体，返回 pyplot。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"
    return plt
