"""环境自检：换机器后跑一遍，定位"某个模块打不开/报错"。

检查内容：
  1. Python 与关键库版本；
  2. 四个模型文件能否在**这台机器**上加载（失败会打印完整异常）；
  3. 每个模块做一次最小推理；
  4. 数据与知识库文件是否齐全。

用法：python tools/doctor.py
退出码：0 全部正常；1 有模块失败（输出里能看到具体原因）。
"""
from __future__ import annotations

import sys
import traceback
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, BAD = "[ OK ]", "[FAIL]"
failed = 0


def line(tag: str, msg: str) -> None:
    global failed
    print(f"{tag} {msg}")
    if tag == BAD:
        failed += 1


def versions() -> None:
    print("== 环境 ==")
    print(f"python  {sys.version.split()[0]}  ({sys.executable})")
    dist = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
            "scipy": "scipy", "joblib": "joblib", "flask": "flask"}
    for name, dist_name in dist.items():
        try:
            __import__(name)
            print(f"{name:<8}{pkg_version(dist_name)}")
        except (ImportError, PackageNotFoundError) as exc:
            line(BAD, f"{name} 未安装：{exc}")


def models() -> None:
    import joblib

    print("\n== 模型加载与推理 ==")
    samples = {
        "ticket": ("ticket_classifier.joblib", lambda m: m.predict_proba(["取件码输进去没反应，柜门打不开"])[0]),
        "segment": ("user_segmenter.joblib", None),
        "volume": ("volume_regressor.joblib", None),
    }
    for key, (fname, probe) in samples.items():
        path = ROOT / "models" / fname
        if not path.exists():
            line(BAD, f"{key}: 模型文件不存在 {path}")
            continue
        try:
            model = joblib.load(path)
        except Exception:
            line(BAD, f"{key}: 加载失败")
            traceback.print_exc(limit=3)
            continue
        if probe is None:
            line(OK, f"{key}: 加载成功（{type(model).__name__}）")
            continue
        try:
            out = probe(model)
            line(OK, f"{key}: 加载并推理成功，首类概率 {float(out.max()):.3f}")
        except Exception:
            line(BAD, f"{key}: 加载成功但推理失败")
            traceback.print_exc(limit=3)

    # 模块二 / 三直接复用接口里的逻辑，用最短路径验证
    try:
        import pandas as pd

        seg = joblib.load(ROOT / "models" / "user_segmenter.joblib")
        feat = seg["features"]
        row = pd.DataFrame([[0.0] * len(feat)], columns=feat)
        Xs = seg["scaler"].transform(row)
        seg["kmeans"].predict(Xs)
        line(OK, "segment: 推理成功")
    except Exception:
        line(BAD, "segment: 推理失败")
        traceback.print_exc(limit=3)

    try:
        import numpy as np

        vol = joblib.load(ROOT / "models" / "volume_regressor.joblib")
        vol["pipeline"].predict(np.zeros((1, len(vol["features"]))))
        line(OK, "volume: 推理成功")
    except Exception:
        line(BAD, "volume: 推理失败")
        traceback.print_exc(limit=3)


def assets() -> None:
    print("\n== 数据与知识库 ==")
    for rel in ("data/knowledge/faq.json", "data/processed/tickets_clean.csv", "README.md"):
        p = ROOT / rel
        line(OK if p.exists() else BAD, f"{rel}: {'存在' if p.exists() else '缺失'}")


def main() -> None:
    versions()
    models()
    assets()
    print()
    if failed:
        print(f"共 {failed} 项失败。若是换机器后出现的，先运行 `python src/run_all.py` 用本机环境重训模型；")
        print("仍失败再把上面的完整输出（含报错堆栈）发回来。")
        sys.exit(1)
    print("全部正常。")


if __name__ == "__main__":
    main()
