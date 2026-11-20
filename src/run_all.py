"""一键执行：生成数据 → 清洗 → 训练四个模块 → 系统评估。

用法：
    python src/run_all.py            # 全流程
    python src/run_all.py --skip-gen # 复用已有数据，只重训与评估
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(module: str, title: str, argv=None) -> None:
    print(f"\n{'=' * 60}\n▶ {title}\n{'=' * 60}")
    t0 = time.perf_counter()
    mod = importlib.import_module(module)
    mod.main() if argv is None else mod.main(argv)
    print(f"✓ {title} 完成，用时 {time.perf_counter() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gen", action="store_true", help="跳过数据生成")
    args = ap.parse_args()

    if not args.skip_gen:
        run("generate_data", "生成合成数据")
    run("preprocess", "数据清洗与特征工程")
    run("module1_ticket", "模块一：工单分类")
    run("module2_cluster", "模块二：用户分群")
    run("module3_regress", "模块三：到件量预测")
    run("eval_qa", "模块四：政策问答评测", argv=[])
    run("evaluate", "系统级评估与失败案例")

    print("\n全部完成。启动系统：python src/app.py --port 7860")


if __name__ == "__main__":
    main()
