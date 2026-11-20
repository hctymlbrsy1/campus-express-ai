"""文档/产物一致性检查：给"禁止手抄"装一个可执行的保险丝。

检查三件事：
  1. `run_meta.json` 记录的 raw / processed 数据哈希，是否与当前文件一致
     （不一致 = 改了数据但没重跑，所有文档数字都已失效）；
  2. `summary.md` 是否早于 `run_meta.json`（早于说明产物未随数据刷新）；
  3. `docs/03` 与 `README` 里的指标数字，是否都能在 `outputs/metrics/` 中找到原值。

用法：python tools/check_docs.py        # 退出码 0=一致，1=有问题
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "outputs" / "metrics" / "run_meta.json"


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    problems: list[str] = []
    if not META.exists():
        print("[FAIL] 缺少 run_meta.json，先运行 python src/run_all.py")
        return 1

    meta = json.loads(META.read_text(encoding="utf-8"))
    gen_at = datetime.strptime(meta["generated_at"], "%Y-%m-%d %H:%M:%S")

    # 1) 哈希核对（raw + processed）
    for key, folder in (("data_sha256", "data/processed"), ("raw_sha256", "data/raw")):
        for name, recorded in meta.get(key, {}).items():
            p = ROOT / name
            if not p.exists():
                problems.append(f"{folder}/{name} 已不存在（run_meta 记录过）")
            elif h(p) != recorded:
                problems.append(f"{folder}/{name} 哈希变化：{recorded} → {h(p)}（改了数据未重跑）")

    # 2) 产物新鲜度（summary.md 由 evaluate.py 在 run_meta 之后写出，早于它说明没重跑）
    summary = ROOT / "outputs" / "metrics" / "summary.md"
    if summary.exists() and datetime.fromtimestamp(summary.stat().st_mtime) < gen_at:
        problems.append("summary.md 早于 run_meta（产物未随数据刷新）")

    # 3) docs/03 与 README 里的指标数字，必须能在产物中原样找到（按小数位四舍五入比对）
    #    产物 = outputs/metrics 下的 JSON / CSV / MD（都是脚本生成的）
    nums: list[float] = []
    for p in sorted((ROOT / "outputs" / "metrics").iterdir()):
        if p.suffix not in (".json", ".csv", ".md"):
            continue
        try:
            txt = p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        nums += [float(x) for x in re.findall(r"-?\d+\.\d+", txt)]

    for rel in ("docs/03_失败案例分析.md", "README.md"):
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}", " ", text)      # 去掉 IP，避免 127.0.0.1 被当成指标
        text = re.sub(r"https?://\S+", " ", text)                  # 去掉徽章/链接 URL
        # 去掉版本号（Python 3.10+、scikit-learn 1.5 等），它们不是指标
        text = re.sub(r"(?:Python|python|scikit-learn|Flask|NumPy|numpy|pandas|scipy|joblib)\s*[≥>=]*\s*\d+(?:\.\d+)*\+?", " ", text)
        text = re.sub(r"\b\d+\.\d+\+", " ", text)
        tokens = set(re.findall(r"\d+\.\d{2,4}", text))
        tokens |= set(re.findall(r"\b\d{2,3}\.\d(?![0-9])", text))
        missing = []
        for t in sorted(tokens):
            dec = len(t.split(".")[1])
            v = float(t)
            tol = 10 ** (-dec) / 2 + 1e-9
            if not any(abs(abs(n) - v) <= tol for n in nums):
                missing.append(t)
        if missing:
            problems.append(f"{rel} 有 {len(missing)} 个数字在产物中找不到：{missing[:8]}")

    if problems:
        print("[FAIL] 发现 %d 处不一致：" % len(problems))
        for x in problems:
            print("  -", x)
        return 1

    print(f"[OK] 文档与产物一致（统计时点 {meta['generated_at']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
