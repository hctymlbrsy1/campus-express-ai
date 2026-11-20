/* 校园快递驿站智能服务系统 —— 前端交互 */
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

const FEATURE_CN = {
  weekly_pickups: "周取件次数",
  avg_dwell_hours: "平均滞留时长 (h)",
  night_ratio: "夜间取件占比",
  weekend_ratio: "周末取件占比",
  proxy_ratio: "代取占比",
  large_item_ratio: "大件占比",
  complaint_cnt: "投诉次数",
  avg_weight_kg: "平均包裹重量 (kg)",
};

let META = null;

/* ---------------- 导航 ---------------- */
function showView(name) {
  const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
  if (!btn) return;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  btn.classList.add("active");
  $("view-" + name).classList.add("active");
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    showView(btn.dataset.view);
    history.replaceState(null, "", "#" + btn.dataset.view);
  });
});

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `请求失败 (${res.status})`);
  return data;
}

/* ---------------- 启动检查 ---------------- */
async function boot() {
  try {
    const h = await api("/api/health");
    const ok = Object.values(h.models).every(Boolean);
    $("status-dot").className = "dot " + (ok ? "ok" : "bad");
    $("status-text").textContent = ok
      ? `四个模块已加载 · 问答 ${h.qa_mode === "llm" ? "在线大模型" : "离线降级"}`
      : "部分模型缺失，请先运行训练脚本";
  } catch (e) {
    $("status-dot").className = "dot bad";
    $("status-text").textContent = "服务未就绪";
  }

  try {
    const s = await api("/api/stats");
    $("s-tickets").textContent = s.tickets ?? "–";
    $("s-acc").textContent = s.classify_acc != null ? (s.classify_acc * 100).toFixed(1) + "%" : "–";
    $("s-acc-sub").textContent = s.classify_best ? `${s.classify_best} · macro-F1 ${fmt(s.classify_f1, 3)}` : "–";
    $("s-sil").textContent = fmt(s.silhouette, 3);
    $("s-k").textContent = s.best_k ?? "–";
    $("s-rmse").textContent = s.rmse != null ? s.rmse + " 件" : "–";
    $("s-r2").textContent = fmt(s.r2, 3);
    renderMetrics(s);
  } catch (e) {
    $("metrics-table").innerHTML = `<p class="muted">${e.message}</p>`;
  }

  try {
    META = await api("/api/meta");
    buildSliders(META.user_features, META.feature_order);
    renderProfiles(META.profiles);
    $("f-lag").textContent = `前1日 ${META.last_volume.lag1}，前7日 ${META.last_volume.lag7}，近7日均值 ${META.last_volume.roll7.toFixed(1)}（${META.last_volume.date}）`;
  } catch (e) {
    $("profiles").innerHTML = `<p class="muted">${e.message}</p>`;
  }
}

function renderMetrics(s) {
  const rows = [
    ["模块一：工单分类", s.classify_best || "–", `准确率 ${fmt(s.classify_acc, 3)} · macro-F1 ${fmt(s.classify_f1, 3)}`, `${s.classes} 类`],
    ["模块二：用户分群", "KMeans + PCA", `轮廓系数 ${fmt(s.silhouette, 3)}（肘部法 + 轮廓系数选 K）`, `K=${s.best_k}`],
    ["模块三：到件量预测", s.reg_best || "–", `RMSE ${s.rmse} 件 · R² ${fmt(s.r2, 3)}`, "时间序列切分"],
    ["模块四：政策问答", "检索增强 + 提示策略", s.qa_mode ? "已完成在线对比" : "离线降级（待配置 API Key）", s.qa_mode ? `${s.qa_calls || 0} 次 LLM 调用` : `${s.qa_calls || 0} 次降级调用 · 0 次 LLM`],
  ];
  $("metrics-table").innerHTML = `<table><thead><tr><th>模块</th><th>方案</th><th>关键指标</th><th>备注</th></tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

/* ---------------- 模块一 ---------------- */
const ticketInput = $("ticket-text");
ticketInput.addEventListener("input", () => ($("ticket-len").textContent = `${ticketInput.value.length} 字`));
document.querySelectorAll("#view-classify .chip").forEach((c) =>
  c.addEventListener("click", () => {
    ticketInput.value = c.dataset.text;
    ticketInput.dispatchEvent(new Event("input"));
  })
);

$("btn-classify").addEventListener("click", async () => {
  const text = ticketInput.value.trim();
  if (!text) return alert("请先输入工单内容");
  const btn = $("btn-classify");
  btn.disabled = true;
  try {
    const r = await api("/api/classify", { text });
    $("classify-result").classList.remove("hidden");
    $("c-label").textContent = r.label_cn;
    $("c-priority").textContent = "优先级：" + r.priority;
    $("c-priority").className = "badge " + (r.priority === "高" ? "high" : "normal");
    $("c-conf-fill").style.width = (r.confidence * 100).toFixed(1) + "%";
    $("c-conf-text").textContent =
      `置信度 ${(r.confidence * 100).toFixed(1)}%` +
      (r.truncated ? " · 文本超长已截断" : "") +
      (r.mode === "rule-fallback" ? " · 模型不可用，已降级为关键词规则" : "");
    $("c-top3").innerHTML = (r.top3 || [])
      .map(
        (t) => `<div class="line"><span>${t.name}</span><span class="bar"><i style="width:${(t.prob * 100).toFixed(1)}%"></i></span><span>${(t.prob * 100).toFixed(1)}%</span></div>`
      )
      .join("");
    $("c-suggestion").textContent = r.suggestion;
    $("c-latency").textContent = r.latency_ms + " ms";
  } catch (e) {
    $("classify-result").classList.remove("hidden");
    $("c-label").textContent = "请求失败";
    $("c-conf-text").textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});

/* ---------------- 模块二 ---------------- */
function buildSliders(features, order) {
  const box = $("sliders");
  const keys = order && order.length ? order.filter((k) => features[k]) : Object.keys(features);
  box.innerHTML = keys
    .map((k) => {
      const r = features[k];
      const step = r.max <= 1 ? 0.01 : r.max <= 10 ? 0.1 : 1;
      return `<label>${FEATURE_CN[k] || k} <span class="val" id="val-${k}">${r.mean}</span>
        <input type="range" id="sl-${k}" min="${r.min}" max="${r.max}" step="${step}" value="${r.mean}" />
      </label>`;
    })
    .join("");
  keys.forEach((k) => {
    $("sl-" + k).addEventListener("input", (e) => ($("val-" + k).textContent = e.target.value));
  });
}

$("btn-segment").addEventListener("click", async () => {
  if (!META) return;
  const payload = {};
  Object.keys(META.user_features).forEach((k) => (payload[k] = parseFloat($("sl-" + k).value)));
  const btn = $("btn-segment");
  btn.disabled = true;
  try {
    const r = await api("/api/segment", payload);
    const same = META.profiles.find((p) => p.cluster === r.cluster) || {};
    $("segment-result").innerHTML = `
      <div class="result-head"><div><span class="muted">所属群体</span><h3>${r.name}</h3></div>
        <span class="badge normal">簇 C${r.cluster}</span></div>
      <div class="kv"><span>与簇中心距离</span><b>${r.distance}</b></div>
      <div class="kv"><span>PCA 坐标</span><b>(${r.pc[0]}, ${r.pc[1]})</b></div>
      <div class="kv"><span>群体规模</span><b>${same["人数"] ?? "–"} 人（${same["占比"] ?? "–"}%）</b></div>
      <div class="kv"><span>触达策略</span><b>${r.strategy || "–"}</b></div>
      <p class="muted">距离越小说明该用户越接近群体典型画像；距离偏大时应结合人工判断，避免机械分组。</p>`;
  } catch (e) {
    $("segment-result").innerHTML = `<p class="muted">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

function renderProfiles(profiles) {
  if (!profiles || !profiles.length) return;
  const cols = ["群体名称", "人数", "占比", "周取件次数", "平均滞留时长(h)", "夜间取件占比",
                "周末取件占比", "代取占比", "大件占比", "投诉次数", "平均包裹重量(kg)"];
  $("profiles").innerHTML =
    `<table><thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${profiles
      .map((p) => `<tr>${cols.map((c) => `<td>${p[c] ?? "–"}</td>`).join("")}</tr>`)
      .join("")}</tbody></table>`;
}

/* ---------------- 模块三 ---------------- */
["promo", "temp", "rain"].forEach((k) =>
  $("f-" + k).addEventListener("input", (e) => ($("v-" + k).textContent = e.target.value))
);

$("btn-forecast").addEventListener("click", async () => {
  if (!META) return;
  const dow = parseInt($("f-dow").value, 10);
  const payload = {
    dow,
    is_weekend: dow >= 5 ? 1 : 0,
    is_holiday: $("f-holiday").checked ? 1 : 0,
    is_vacation: $("f-vacation").checked ? 1 : 0,
    promo_index: parseFloat($("f-promo").value),
    temp_c: parseFloat($("f-temp").value),
    rain_mm: parseFloat($("f-rain").value),
    lag1_volume: META.last_volume.lag1,
    lag7_volume: META.last_volume.lag7,
    roll7_mean: META.last_volume.roll7,
    month: parseInt($("f-month").value, 10),
  };
  const btn = $("btn-forecast");
  btn.disabled = true;
  try {
    const r = await api("/api/forecast", payload);
    const diff = r.volume - r.baseline_roll7;
    $("forecast-result").innerHTML = `
      <span class="muted">预测到件量</span>
      <div><span class="big-number">${r.volume}</span><span class="big-unit">件</span></div>
      <div class="kv"><span>对比近 7 日均值</span><b style="color:${diff >= 0 ? "#b42318" : "#12805c"}">${diff >= 0 ? "+" : ""}${diff.toFixed(1)} 件</b></div>
      <div class="kv"><span>双 11 情景（促销=1.0）</span><b>${r.scenario_promo} 件（+${r.delta}）</b></div>
      <div class="kv"><span>响应时间</span><b>${r.latency_ms} ms</b></div>
      <p class="muted">${r.volume >= 500 ? "预计高峰，建议提前增派 1–2 名兼职并启用临时货架。" : "运力充足，按常规排班即可。"}</p>`;
  } catch (e) {
    $("forecast-result").innerHTML = `<p class="muted">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

/* ---------------- 模块四 ---------------- */
document.querySelectorAll("#view-qa .chip").forEach((c) =>
  c.addEventListener("click", () => ($("qa-input").value = c.dataset.q))
);

$("btn-qa").addEventListener("click", async () => {
  const question = $("qa-input").value.trim();
  if (!question) return alert("请先输入问题");
  const strategy = document.querySelector('input[name="strategy"]:checked').value;
  const btn = $("btn-qa");
  btn.disabled = true;
  try {
    const r = await api("/api/qa", { question, strategy });
    $("qa-result").classList.remove("hidden");
    $("qa-answer").textContent = r.answer;
    $("qa-mode").textContent = r.mode === "llm" ? "在线大模型" : r.mode === "offline-fallback" ? "离线降级" : "调用失败降级";
    $("qa-mode").className = "badge " + (r.mode === "llm" ? "normal" : "offline");
    $("qa-strategy").textContent = r.strategy_cn;
    $("qa-latency").textContent = r.latency_ms + " ms";
    $("qa-sources").textContent = (r.sources || []).join(" / ") || "未命中条款";
  } catch (e) {
    $("qa-result").classList.remove("hidden");
    $("qa-answer").textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});

boot();

/* 支持通过 URL hash 直接打开某个模块，例如 /#qa */
if (location.hash) showView(location.hash.slice(1));

/* 演示模式：/?demo=classify 或 /?demo=qa，用于答辩现场一键展示 */
const demo = new URLSearchParams(location.search).get("demo");
if (demo === "classify") {
  showView("classify");
  ticketInput.value = "取件码 12345678 在菜鸟驿站柜子上输进去没反应，柜门打不开，急用";
  ticketInput.dispatchEvent(new Event("input"));
  $("btn-classify").click();
} else if (demo === "qa") {
  showView("qa");
  $("qa-input").value = "包裹最多能放几天？超时怎么收费？";
  $("btn-qa").click();
} else if (demo === "forecast") {
  showView("forecast");
  $("f-promo").value = "0.55";
  $("f-promo").dispatchEvent(new Event("input"));
  setTimeout(() => $("btn-forecast").click(), 400);
}
