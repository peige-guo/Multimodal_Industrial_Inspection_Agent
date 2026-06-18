"use strict";

const $ = (id) => document.getElementById(id);
const apiBase = () => $("apiBase").value.replace(/\/$/, "");

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function checkHealth() {
  const dot = $("apiStatus");
  try {
    const res = await fetch(`${apiBase()}/health`);
    const ok = res.ok && (await res.json()).status === "ok";
    dot.className = "status-dot " + (ok ? "ok" : "bad");
    dot.title = ok ? "Backend healthy" : "Backend error";
  } catch {
    dot.className = "status-dot bad";
    dot.title = "Backend unreachable";
  }
}

$("image").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const wrap = $("imagePreviewWrap");
  if (!file) { wrap.hidden = true; return; }
  $("imagePreview").src = URL.createObjectURL(file);
  wrap.hidden = false;
});

$("inspectForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("submitBtn");
  const err = $("errorMsg");
  err.hidden = true;

  const fd = new FormData();
  fd.append("image", $("image").files[0]);
  fd.append("standard", $("standard").files[0]);
  if ($("sensor").files[0]) fd.append("sensor_csv", $("sensor").files[0]);
  if ($("objectName").value.trim()) fd.append("object_name", $("objectName").value.trim());
  const hints = $("visionHints").value.trim();
  if (hints) {
    try { JSON.parse(hints); } catch {
      err.textContent = "Vision hints must be valid JSON.";
      err.hidden = false;
      return;
    }
    fd.append("vision_hints", hints);
  }

  btn.disabled = true;
  btn.textContent = "Running inspection...";
  try {
    const res = await fetch(`${apiBase()}/api/inspect`, { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
    renderReport(body);
  } catch (ex) {
    err.textContent = ex.message;
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run inspection";
  }
});

function renderReport(r) {
  $("empty").hidden = true;
  const el = $("report");
  el.hidden = false;

  const sev = r.severity_level;
  const evidence = (r.standard_evidence || []).length
    ? r.standard_evidence
        .map(
          (e) =>
            `<div class="evidence-item"><span class="clause">${escapeHtml(e.clause_id || "clause")}</span>${escapeHtml(e.title || "")}<div>${escapeHtml(e.text)}</div></div>`
        )
        .join("")
    : `<p class="muted">No matching standard clause found.</p>`;

  const reasons = (r.human_review_reasons || []).length
    ? r.human_review_reasons.map((x) => `<div class="reason-item">${escapeHtml(x)}</div>`).join("")
    : `<p class="muted">No review reasons.</p>`;

  const sensors = (r.sensor_readings || []).length
    ? `<table class="sensor-table"><thead><tr><th>Sensor</th><th>Value</th><th>Unit</th></tr></thead><tbody>${r.sensor_readings
        .map(
          (s) =>
            `<tr><td>${escapeHtml(s.name)}</td><td>${escapeHtml(s.value)}</td><td>${escapeHtml(s.unit || "")}</td></tr>`
        )
        .join("")}</tbody></table>`
    : `<p class="muted">No sensor data provided.</p>`;

  el.innerHTML = `
    <div class="report-header">
      <div>
        <strong>${escapeHtml(r.object_name)}</strong>
        <div class="report-id">${escapeHtml(r.inspection_id)}</div>
      </div>
      <div class="report-id">${escapeHtml(r.inspection_time)}</div>
    </div>

    <div class="badges">
      <span class="badge sev-${escapeHtml(sev)}">${escapeHtml(sev)}</span>
      <span class="badge action">${escapeHtml(r.recommended_action)}</span>
      <span class="badge ${r.requires_human_review ? "review-yes" : "review-no"}">
        ${r.requires_human_review ? "human review" : "auto-cleared"}
      </span>
    </div>

    <h3>Defect</h3>
    <dl class="kv">
      <dt>Type</dt><dd>${escapeHtml(r.defect_type)}</dd>
      <dt>Location</dt><dd>${escapeHtml(r.defect_location || "n/a")}</dd>
      <dt>Confidence</dt><dd>${(r.confidence * 100).toFixed(0)}%</dd>
      <dt>Inputs</dt><dd>${escapeHtml(r.input_data_summary)}</dd>
    </dl>

    <h3>Risk explanation</h3>
    <p>${escapeHtml(r.risk_explanation)}</p>

    <h3>Standard evidence</h3>
    ${evidence}

    <h3>Sensor readings</h3>
    ${sensors}

    <h3>Human review</h3>
    ${reasons}
  `;
}

$("apiBase").addEventListener("change", checkHealth);
checkHealth();
