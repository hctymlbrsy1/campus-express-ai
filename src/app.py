"""模块五：系统集成 —— Flask Web 服务

四个算法模块统一封装为 JSON 接口，前端为单页应用（templates/index.html）。
所有接口都做入参校验、超长截断、异常兜底，并记录响应时间与大模型降级状态。
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from common import DATA_PROC, METRICS, MODELS, TEXT_MAX_LEN, clip_text, ensure_dirs, normalize_text
from module1_ticket import LABEL_CN, rule_predict
from module4_qa import QAEngine, STRATEGIES

ensure_dirs()
app = Flask(__name__, template_folder="templates", static_folder="static")
app.json.ensure_ascii = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(Path(__file__).resolve().parents[1] / "logs" / "app.log", encoding="utf-8")],
)
log = logging.getLogger("express-ai")

# ---------------------------------------------------------------- 模型加载
BUNDLE: dict = {}
BUNDLE_ERRORS: dict[str, str] = {}   # 记录加载失败原因，供 /api/health 与排查脚本使用


def _load(name: str, loader) -> None:
    try:
        BUNDLE[name] = loader()
    except Exception as exc:  # 版本不兼容、文件缺失等都在这里暴露
        BUNDLE_ERRORS[name] = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("%s 加载失败: %s", name, exc)


def load_models() -> None:
    _load("ticket", lambda: joblib.load(MODELS / "ticket_classifier.joblib"))
    _load("segment", lambda: joblib.load(MODELS / "user_segmenter.joblib"))
    _load("volume", lambda: joblib.load(MODELS / "volume_regressor.joblib"))
    _load("qa", QAEngine)


def _metric(name: str) -> dict:
    p = METRICS / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 页面
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "models": {k: (k in BUNDLE) for k in ("ticket", "segment", "volume", "qa")},
            "errors": BUNDLE_ERRORS,
            "qa_mode": "llm" if getattr(BUNDLE.get("qa"), "api_key", "") else "offline-fallback",
        }
    )


@app.get("/api/stats")
def stats():
    m1, m2, m3, m4 = (_metric(n) for n in ("module1.json", "module2.json", "module3.json", "module4.json"))
    best1 = m1.get("models", {}).get(m1.get("best", ""), {})
    best3 = m3.get("models", {}).get(m3.get("best", ""), {})
    return jsonify(
        {
            "tickets": m1.get("dataset", {}).get("rows"),
            "classes": m1.get("dataset", {}).get("classes"),
            "classify_acc": best1.get("accuracy"),
            "classify_f1": best1.get("f1_macro"),
            "classify_best": best1.get("name"),
            "silhouette": m2.get("silhouette"),
            "best_k": m2.get("best_k"),
            "ari": m2.get("ari_vs_true_segment"),
            "rmse": best3.get("RMSE"),
            "r2": best3.get("R2"),
            "reg_best": best3.get("name"),
            # 只有当"产物里有真实对比"且"本机配了 Key"时才显示在线，否则给出降级提示
            "qa_mode": bool(m4.get("valid_for_conclusion")) and bool(getattr(BUNDLE.get("qa"), "api_key", "")),
            "qa_calls": m4.get("n_calls"),
        }
    )


@app.get("/api/meta")
def meta():
    """前端滑块默认值与范围。"""
    users = pd.read_csv(DATA_PROC / "users_clean.csv", encoding="utf-8-sig")
    seg = BUNDLE.get("segment", {})
    feats = seg.get("features", [])
    ranges = {
        f: {
            "min": round(float(users[f].min()), 2),
            "max": round(float(users[f].max()), 2),
            "mean": round(float(users[f].mean()), 2),
        }
        for f in feats
    }
    vol = pd.read_csv(DATA_PROC / "volume_clean.csv", encoding="utf-8-sig")
    return jsonify(
        {
            "user_features": ranges,
            "feature_order": feats,
            "labels": LABEL_CN,
            "strategies": STRATEGIES,
            "last_volume": {
                "date": str(vol["date"].iloc[-1]),
                "lag1": float(vol["lag1_volume"].iloc[-1]),
                "lag7": float(vol["lag7_volume"].iloc[-1]),
                "roll7": float(vol["roll7_mean"].iloc[-1]),
            },
            "profiles": _metric("module2.json").get("profiles", []),
            "strategies_by_cluster": _metric("module2.json").get("strategies", {}),
        }
    )


# ---------------------------------------------------------------- 模块一
@app.post("/api/classify")
def classify():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    raw = str(data.get("text", ""))
    text = normalize_text(raw)
    if not text:
        return jsonify({"error": "输入为空", "hint": "请输入一条工单描述，例如：取件码输进去没反应"}), 400

    text, truncated = clip_text(text, TEXT_MAX_LEN)
    model = BUNDLE.get("ticket")
    mode, note = "model", ""

    top3: list[dict] = []
    if model is None:
        # 模型没加载：退到关键词规则，至少保证演示可用
        label, conf = rule_predict(text)
        mode = "rule-fallback"
        note = f"模型未加载（{BUNDLE_ERRORS.get('ticket', '未知原因')}），已改用关键词规则"
    else:
        try:
            proba = model.predict_proba([text])[0]
            classes = list(model.named_steps["clf"].classes_)
            order = np.argsort(-proba)
            label = classes[order[0]]
            conf = float(proba[order[0]])
            top3 = [{"label": classes[i], "name": LABEL_CN.get(classes[i], classes[i]),
                     "prob": round(float(proba[i]), 4)} for i in order[:3]]
        except Exception as exc:
            # 跨机器最常见的失败：训练与运行环境的 sklearn / numpy 版本不一致
            log.exception("工单分类推理失败，改用规则兜底")
            label, conf = rule_predict(text)
            mode = "rule-fallback"
            note = f"模型推理失败（{type(exc).__name__}），已改用关键词规则；详见 logs/app.log"

    # 工单优先级规则：丢件/赔付类自动升级为高优先级
    priority = "高" if label in ("lost", "compensate") else "普通"

    resp = {
        "text": text,
        "truncated": truncated,
        "label": label,
        "label_cn": LABEL_CN.get(label, label),
        "confidence": round(float(conf), 4),
        "top3": top3,
        "priority": priority,
        "suggestion": "建议转人工核实监控与理赔材料" if priority == "高" else "建议按标准话术自动回复",
        "mode": mode,
        "note": note,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    log.info("classify len=%d pred=%s conf=%.3f %.1fms", len(text), resp["label"], resp["confidence"], resp["latency_ms"])
    return jsonify(resp)


# ---------------------------------------------------------------- 模块二
@app.post("/api/segment")
def segment():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    bundle = BUNDLE.get("segment")
    if bundle is None:
        return jsonify({"error": "模型未加载", "hint": "请先运行 src/module2_cluster.py"}), 503

    feats = bundle["features"]
    try:
        vec = np.array([[float(data.get(f, 0)) for f in feats]], dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "特征值必须为数值"}), 400
    if not np.isfinite(vec).all():
        return jsonify({"error": "特征值包含非法数字"}), 400

    vec = np.clip(vec, 0, None)
    Xs = bundle["scaler"].transform(vec)
    cid = int(bundle["kmeans"].predict(Xs)[0])
    name = bundle["names"].get(cid, f"群体{cid}")
    dist = float(np.linalg.norm(Xs - bundle["kmeans"].cluster_centers_[cid]))
    Z = bundle["pca"].transform(Xs)[0]

    profiles = _metric("module2.json").get("profiles", [])
    prof = next((p for p in profiles if p.get("cluster") == cid), {})
    strategies = _metric("module2.json").get("strategies", {})

    resp = {
        "cluster": cid,
        "name": name,
        "distance": round(dist, 3),
        "pc": [round(float(Z[0]), 3), round(float(Z[1]), 3)],
        "profile": prof,
        "strategy": strategies.get(name, ""),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    log.info("segment cluster=%s dist=%.2f %.1fms", name, dist, resp["latency_ms"])
    return jsonify(resp)


# ---------------------------------------------------------------- 模块三
@app.post("/api/forecast")
def forecast():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    bundle = BUNDLE.get("volume")
    if bundle is None:
        return jsonify({"error": "模型未加载", "hint": "请先运行 src/module3_regress.py"}), 503

    feats = bundle["features"]
    try:
        row = {
            "dow": float(data.get("dow", 0)),
            "is_weekend": float(data.get("is_weekend", 0)),
            "is_holiday": float(data.get("is_holiday", 0)),
            "is_vacation": float(data.get("is_vacation", 0)),
            "promo_index": float(data.get("promo_index", 0)),
            "temp_c": float(data.get("temp_c", 20)),
            "rain_mm": float(data.get("rain_mm", 0)),
            "lag1_volume": float(data.get("lag1_volume", 0)),
            "lag7_volume": float(data.get("lag7_volume", 0)),
            "roll7_mean": float(data.get("roll7_mean", 0)),
            "month": float(data.get("month", 1)),
        }
    except (TypeError, ValueError):
        return jsonify({"error": "预测特征必须为数值"}), 400

    def to_matrix(r: dict) -> pd.DataFrame:
        X = pd.DataFrame([r])
        for i in range(1, 7):
            X[f"dow_{i}"] = float(int(r["dow"]) == i)
        return X[feats]

    pipe = bundle["pipeline"]
    pred = float(pipe.predict(to_matrix(row))[0])
    # 情景对比：把促销强度拉到双 11 水平，看增量
    pred_promo = float(pipe.predict(to_matrix(dict(row, promo_index=1.0)))[0])

    resp = {
        "volume": int(round(pred)),
        "scenario_promo": int(round(pred_promo)),
        "delta": int(round(pred_promo - pred)),
        "baseline_roll7": round(float(row.get("roll7_mean", 0)), 1),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    log.info("forecast pred=%d %.1fms", resp["volume"], resp["latency_ms"])
    return jsonify(resp)


# ---------------------------------------------------------------- 模块四
@app.post("/api/qa")
def qa():
    eng = BUNDLE.get("qa")
    if eng is None:
        return jsonify({"error": "问答引擎未初始化"}), 503
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", ""))
    strategy = data.get("strategy", "cot")
    if strategy not in STRATEGIES:
        return jsonify({"error": f"不支持的策略: {strategy}"}), 400
    if not question.strip():
        return jsonify({"error": "问题不能为空"}), 400

    r = eng.ask(question, strategy=strategy)
    r["strategy_cn"] = STRATEGIES[strategy]
    log.info("qa strategy=%s mode=%s %.1fms", strategy, r["mode"], r["latency_ms"])
    return jsonify(r)


# ---------------------------------------------------------------- 异常兜底
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "接口不存在"}), 404


@app.errorhandler(500)
def server_error(exc):
    log.exception("服务异常")
    return jsonify({"error": "服务内部错误", "detail": str(exc)[:200]}), 500


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    load_models()
    print("模型状态:", {k: v is not None for k, v in BUNDLE.items()})
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
