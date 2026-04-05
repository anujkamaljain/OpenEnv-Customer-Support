"""FastAPI server exposing the CustomerSupportEnv over HTTP.

Endpoints:
    GET  /              — browser debug UI (HTML)
    GET  /docs          — interactive OpenAPI (Swagger UI, default theme)
    POST /reset         — start a new episode
    POST /step          — apply an action
    GET  /state         — read current state without advancing
    POST /close         — release episode resources
    POST /inference     — run the full LLM inference loop
    GET  /health        — liveness probe (JSON)
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from env.environment import CustomerSupportEnv

app = FastAPI(
    title="OpenEnv Customer Support",
    version="0.1.0",
    description=(
        "Real-world customer support ticket triage environment. "
        "An agent learns to classify, prioritize, and route support tickets."
    ),
)

_env = CustomerSupportEnv()


# ── request / response schemas ───────────────────────────────────────────────


class ResetRequest(BaseModel):
    """POST /reset body. Valid ``difficulty`` values are easy, medium, hard, or omitted for all pools."""

    seed: int = Field(default=0, description="Deterministic ticket index within the selected pool.")
    difficulty: Literal["easy", "medium", "hard"] | None = Field(
        default=None,
        description='Ticket pool filter, or omit / null for all difficulties.',
    )


class StepRequest(BaseModel):
    action: dict[str, Any]


class EnvResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class InferenceResponse(BaseModel):
    stdout: str
    score: float
    success: bool


# ── helpers ──────────────────────────────────────────────────────────────────


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


_DEBUG_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Customer Support Command Center</title>
  <style>
    :root {
      --bg: #141517;
      --surface: #25262b;
      --border: #373a40;
      --text: #e9ecef;
      --muted: #868e96;
      --accent: #4c6ef5;
      --exec: #e8590c;
      --exec-hover: #fd7e14;
      --hint-bg: #1b4332;
      --hint-border: #2b8a3e;
      --card-blue: #1864ab;
      --card-orange: #d9480f;
      --card-green: #2b8a3e;
      --card-grey: #495057;
      --danger: #fa5252;
      --mono: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
    }
    * { box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .wrap { max-width: 56rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
    h1 { font-size: 1.35rem; font-weight: 700; margin: 0 0 0.25rem; letter-spacing: -0.02em; }
    .sub { color: var(--muted); font-size: 0.875rem; margin: 0 0 1.25rem; }
    .hint-bar {
      background: var(--hint-bg);
      border: 1px solid var(--hint-border);
      color: #b2f2bb;
      font-size: 0.8rem;
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      margin-bottom: 1rem;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.15rem;
      margin-bottom: 1rem;
    }
    .panel h2 {
      margin: 0 0 0.75rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
    }
    label { font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 0.3rem; }
    label .hint { font-weight: 400; color: #5c636a; }
    select, input[type="text"], input[type="number"], textarea {
      width: 100%;
      padding: 0.5rem 0.55rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #1a1b1e;
      color: var(--text);
      font-size: 0.85rem;
    }
    select {
      cursor: pointer;
      appearance: auto;
      min-height: 2.25rem;
    }
    textarea { font-family: var(--mono); font-size: 0.8rem; min-height: 4.5rem; resize: vertical; }
    /* Dark scrollbars — match theme (textarea JSON preview, pre blocks, timeline) */
    textarea,
    pre {
      scrollbar-width: thin;
      scrollbar-color: #5c636a #141517;
    }
    textarea::-webkit-scrollbar,
    pre::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    textarea::-webkit-scrollbar-corner,
    pre::-webkit-scrollbar-corner {
      background: #141517;
    }
    textarea::-webkit-scrollbar-track,
    pre::-webkit-scrollbar-track {
      background: #141517;
      border-radius: 4px;
    }
    textarea::-webkit-scrollbar-thumb,
    pre::-webkit-scrollbar-thumb {
      background: #5c636a;
      border-radius: 4px;
      border: 2px solid #141517;
    }
    textarea::-webkit-scrollbar-thumb:hover,
    pre::-webkit-scrollbar-thumb:hover {
      background: #868e96;
    }
    /* Number inputs: drop default light steppers (seed still editable) */
    input[type="number"] {
      -moz-appearance: textfield;
      appearance: textfield;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    .form-grid {
      display: grid;
      gap: 0.75rem;
      margin-top: 0.75rem;
    }
    @media (min-width: 560px) {
      .form-grid.cols-2 { grid-template-columns: 1fr 1fr; }
    }
    .field-group { display: none; }
    .field-group.active { display: block; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
    button {
      border: none;
      padding: 0.55rem 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    button.exec { background: var(--exec); color: #fff; }
    button.exec:hover { background: var(--exec-hover); }
    button.secondary { background: #495057; color: #fff; }
    button.secondary:hover { background: #5c636a; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.65rem;
    }
    @media (min-width: 720px) { .metrics { grid-template-columns: repeat(4, 1fr); } }
    .metric {
      border-radius: 8px;
      padding: 0.75rem 0.85rem;
      color: #fff;
    }
    .metric .m-label { font-size: 0.65rem; text-transform: uppercase; opacity: 0.9; letter-spacing: 0.06em; }
    .metric .m-val { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .metric.blue { background: linear-gradient(135deg, var(--card-blue), #1c7ed6); }
    .metric.orange { background: linear-gradient(135deg, var(--card-orange), #e8590c); }
    .metric.green { background: linear-gradient(135deg, var(--card-green), #37b24d); }
    .metric.grey { background: linear-gradient(135deg, #495057, #6c757d); }
    pre {
      margin: 0;
      padding: 0.65rem 0.75rem;
      background: #1a1b1e;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 0.72rem;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .tag { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.25rem; }
    #status { font-size: 0.8rem; min-height: 1.25rem; margin: 0.5rem 0; }
    #status.err { color: var(--danger); }
    .json-preview label { margin-top: 0.75rem; }
    #actionJson { min-height: 5rem; opacity: 0.92; }
    footer { margin-top: 1.25rem; font-size: 0.72rem; color: var(--muted); }
    footer a { color: var(--accent); }

    /* Current ticket (reference-style card) */
    h2.section-title {
      font-size: 1rem;
      font-weight: 700;
      color: #f8f9fa;
      margin: 0 0 0.5rem;
      letter-spacing: -0.01em;
    }
    .ticket-panel {
      background: #1e1f23 !important;
      border-color: #2c2e33 !important;
      padding: 1rem 1.1rem 1.1rem !important;
    }
    .ticket-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-bottom: 0.85rem;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.28rem 0.65rem;
      border-radius: 5px;
      background: #2f3138;
      border: 1px solid #3d4049;
      font-size: 0.72rem;
      color: #dee2e6;
    }
    .pill kbd {
      font-family: inherit;
      font-weight: 600;
      color: #adb5bd;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .pill span.val { font-weight: 600; color: #fff; }
    .ticket-msg-wrap {
      background: #121214;
      border: 1px solid #2c2e33;
      border-radius: 6px;
      padding: 0.75rem 0.9rem 1rem;
    }
    .ticket-msg-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #868e96;
      margin-bottom: 0.5rem;
    }
    .ticket-msg-body {
      font-size: 0.95rem;
      line-height: 1.55;
      color: #f1f3f5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .ticket-msg-body.empty { color: #5c636a; font-style: italic; }

    /* Action timeline */
    .timeline-panel {
      background: #252830 !important;
      border-color: #3d4150 !important;
    }
    .timeline-scroll {
      max-height: 18rem;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-gutter: stable;
      scrollbar-width: thin;
      scrollbar-color: #5c636a #141517;
    }
    .timeline-scroll::-webkit-scrollbar { width: 8px; }
    .timeline-scroll::-webkit-scrollbar-track { background: #141517; border-radius: 4px; }
    .timeline-scroll::-webkit-scrollbar-thumb {
      background: #5c636a;
      border-radius: 4px;
      border: 2px solid #141517;
    }
    .timeline-scroll::-webkit-scrollbar-thumb:hover { background: #868e96; }
    .timeline-list { display: flex; flex-direction: column; gap: 0.5rem; }
    .timeline-item {
      background: #121214;
      border: 1px solid #2c2e33;
      border-radius: 6px;
      padding: 0.65rem 0.85rem;
      overflow: hidden;
    }
    .timeline-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
    }
    .tl-main { min-width: 0; flex: 1; }
    .tl-step-num {
      font-weight: 700;
      color: #e9ecef;
      font-size: 0.82rem;
    }
    .tl-action-name {
      font-size: 0.82rem;
      color: #ced4da;
      font-weight: 500;
    }
    .tl-reward {
      font-size: 0.85rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      color: #fff;
      flex-shrink: 0;
    }
    .tl-reward.neg { color: #ff8787; }
    .tl-reward.pos { color: #8ce99a; }
    .tl-feedback {
      margin-top: 0.45rem;
      padding-top: 0.45rem;
      border-top: 1px solid #2c2e33;
      font-size: 0.72rem;
      line-height: 1.4;
      color: #868e96;
    }
    .timeline-empty {
      text-align: center;
      padding: 1.25rem 0.75rem;
      color: #868e96;
      font-size: 0.82rem;
      background: #121214;
      border-radius: 6px;
      border: 1px dashed #3d4049;
    }

    .page-head {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.25rem;
    }
    .page-head-text { flex: 1; min-width: 12rem; }
    .page-head-text h1 { margin-bottom: 0.25rem; }
    .page-head-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      align-items: center;
    }
    a.btn-mini {
      display: inline-block;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      text-decoration: none;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #fdfd96;
      color: #111;
      border: none;
      cursor: pointer;
      font-family: inherit;
      line-height: 1.2;
    }
    a.btn-mini:hover { filter: brightness(1.06); }
    a.btn-mini:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page-head">
      <div class="page-head-text">
        <h1>Customer Support Command Center</h1>
        <p class="sub">Classify → route → respond / escalate / request info → resolve ·
          <a href="/health" style="color:var(--accent)">/health</a></p>
      </div>
      <div class="page-head-actions">
        <a class="btn-mini" href="/docs" target="_blank" rel="noopener noreferrer">Docs</a>
      </div>
    </header>

    <div class="metrics panel" style="padding:0.85rem;margin-bottom:1rem;background:#1a1b1e;border-style:dashed">
      <div class="metric blue">
        <div class="m-label">Last step reward</div>
        <div class="m-val" id="mLastReward">—</div>
      </div>
      <div class="metric orange">
        <div class="m-label">Total reward</div>
        <div class="m-val" id="mTotalReward">—</div>
      </div>
      <div class="metric green">
        <div class="m-label">Score</div>
        <div class="m-val" id="mScore">—</div>
      </div>
      <div class="metric grey">
        <div class="m-label">Status</div>
        <div class="m-val" id="mStatus" style="font-size:1rem">—</div>
      </div>
    </div>

    <h2 class="section-title">Current Ticket</h2>
    <div class="panel ticket-panel">
      <div class="ticket-pills" id="ticketPills"></div>
      <div class="ticket-msg-wrap">
        <div class="ticket-msg-label">Customer Message</div>
        <div class="ticket-msg-body empty" id="ticketBody">Reset or load state to show the active ticket.</div>
      </div>
    </div>

    <div class="panel">
      <div class="hint-bar">
        Follow the env phase: start with <strong>classify</strong>, then <strong>route</strong>, then optional
        <strong>respond</strong> / <strong>escalate</strong> / <strong>request_info</strong>, then <strong>resolve</strong>.
        Use <strong>Get state</strong> to see <code>available_actions</code>.
        <strong>Ticket difficulty:</strong> reset with <code>seed=0</code> and no filter always picks the first ticket in the bank (easy).
        Choose <strong>medium</strong> / <strong>hard</strong> below, or with “any” use <strong>seed ≥ 5</strong> to leave the easy pool (5 easy tickets: seeds 0–4).
      </div>
      <h2>Step-by-step action</h2>
      <label for="actionType">Action type <span class="hint">(matches POST /step JSON)</span></label>
      <select id="actionType">
        <option value="classify">classify</option>
        <option value="route">route</option>
        <option value="respond">respond</option>
        <option value="escalate">escalate</option>
        <option value="request_info">request_info</option>
        <option value="resolve">resolve</option>
      </select>

      <div id="fgClassify" class="field-group form-grid cols-2">
        <div>
          <label for="clsCategory">Category</label>
          <select id="clsCategory">
            <option value="billing">billing</option>
            <option value="bug_report">bug_report</option>
            <option value="feature_request">feature_request</option>
            <option value="account_access">account_access</option>
            <option value="general_inquiry" selected>general_inquiry</option>
            <option value="cancellation">cancellation</option>
          </select>
        </div>
        <div>
          <label for="clsPriority">Priority</label>
          <select id="clsPriority">
            <option value="low">low</option>
            <option value="medium" selected>medium</option>
            <option value="high">high</option>
            <option value="critical">critical</option>
          </select>
        </div>
      </div>

      <div id="fgRoute" class="field-group form-grid">
        <div>
          <label for="rteDept">Department</label>
          <select id="rteDept">
            <option value="billing">billing</option>
            <option value="technical">technical</option>
            <option value="account">account</option>
            <option value="general" selected>general</option>
          </select>
        </div>
      </div>

      <div id="fgRespond" class="field-group form-grid">
        <div>
          <label for="rspTone">Tone</label>
          <select id="rspTone">
            <option value="formal" selected>formal</option>
            <option value="empathetic">empathetic</option>
            <option value="concise">concise</option>
          </select>
        </div>
        <div style="grid-column:1/-1">
          <label for="rspText">Response text</label>
          <textarea id="rspText" spellcheck="false">Thank you for contacting us. We are reviewing your request.</textarea>
        </div>
      </div>

      <div id="fgEscalate" class="field-group form-grid cols-2">
        <div>
          <label for="escTeam">Target team</label>
          <select id="escTeam">
            <option value="l2_support" selected>l2_support</option>
            <option value="engineering">engineering</option>
            <option value="management">management</option>
          </select>
        </div>
        <div style="grid-column:1/-1">
          <label for="escReason">Reason</label>
          <textarea id="escReason" spellcheck="false">Requires specialist review per policy.</textarea>
        </div>
      </div>

      <div id="fgRequestInfo" class="field-group form-grid">
        <div>
          <label for="reqQ">Question to customer</label>
          <textarea id="reqQ" spellcheck="false">Could you share your account email and approximate time of the issue?</textarea>
        </div>
      </div>

      <div id="fgResolve" class="field-group form-grid">
        <div style="grid-column:1/-1">
          <label for="rsvSummary">Resolution summary</label>
          <textarea id="rsvSummary" spellcheck="false">Issue reviewed and resolved to customer satisfaction.</textarea>
        </div>
        <div>
          <label for="rsvComp">Offered compensation <span class="hint">(optional, empty = omit)</span></label>
          <input type="number" id="rsvComp" step="any" placeholder="e.g. 29.99" />
        </div>
      </div>

      <div class="json-preview">
        <label for="actionJson">Request body preview <code>{"action": …}</code></label>
        <textarea id="actionJson" readonly spellcheck="false"></textarea>
      </div>

      <div class="form-grid cols-2" style="margin-top:0.85rem">
        <div>
          <label for="resetDifficulty">Reset · difficulty <span class="hint">(empty = all pools; seed 0 = first easy)</span></label>
          <select id="resetDifficulty">
            <option value="">Any — seed walks easy → medium → hard</option>
            <option value="easy">easy only</option>
            <option value="medium">medium only</option>
            <option value="hard">hard only</option>
          </select>
        </div>
        <div>
          <label for="resetSeed">Reset · seed</label>
          <input type="number" id="resetSeed" value="0" step="1" />
        </div>
      </div>

      <div class="btn-row">
        <button type="button" class="exec" id="btnStep">Execute step</button>
        <button type="button" class="secondary" id="btnReset">Reset</button>
        <button type="button" class="secondary" id="btnState">Get state</button>
      </div>
    </div>

    <h2 class="section-title">Action Timeline</h2>
    <div class="panel timeline-panel">
      <div class="timeline-scroll">
        <div class="timeline-list" id="timelineList">
          <div class="timeline-empty">No actions yet. Run <strong>Reset</strong>, then <strong>Execute step</strong>.</div>
        </div>
      </div>
    </div>

    <p id="status"></p>

    <div class="panel">
      <div class="tag">observation</div>
      <pre id="outObs">—</pre>
    </div>
    <div class="panel grid" style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
      <div>
        <div class="tag">reward (response)</div>
        <pre id="outReward">—</pre>
      </div>
      <div>
        <div class="tag">done</div>
        <pre id="outDone">—</pre>
      </div>
    </div>
    <div class="panel">
      <div class="tag">info</div>
      <pre id="outInfo">—</pre>
    </div>

    <footer>Built from dropdowns → <code>POST /step</code>. Toggle action type to edit fields.</footer>
  </div>
  <script>
(function () {
  const $ = (id) => document.getElementById(id);
  const status = $("status");
  const outObs = $("outObs");
  const outReward = $("outReward");
  const outDone = $("outDone");
  const outInfo = $("outInfo");
  const mLast = $("mLastReward");
  const mTotal = $("mTotalReward");
  const mScore = $("mScore");
  const mStat = $("mStatus");

  let totalReward = 0;

  function pretty(obj) {
    return JSON.stringify(obj, null, 2);
  }

  function buildAction() {
    const t = $("actionType").value;
    switch (t) {
      case "classify":
        return {
          action_type: "classify",
          category: $("clsCategory").value,
          priority: $("clsPriority").value,
        };
      case "route":
        return { action_type: "route", department: $("rteDept").value };
      case "respond":
        return {
          action_type: "respond",
          response_text: $("rspText").value.trim() || " ",
          tone: $("rspTone").value,
        };
      case "escalate":
        return {
          action_type: "escalate",
          reason: $("escReason").value.trim() || "Escalation.",
          target_team: $("escTeam").value,
        };
      case "request_info":
        return {
          action_type: "request_info",
          question_to_customer: $("reqQ").value.trim() || "Please provide more details.",
        };
      case "resolve": {
        const raw = $("rsvComp").value.trim();
        let comp = null;
        if (raw !== "") {
          const n = Number(raw);
          comp = Number.isFinite(n) ? n : null;
        }
        const o = {
          action_type: "resolve",
          resolution_summary: $("rsvSummary").value.trim() || "Resolved.",
        };
        if (comp !== null) o.offered_compensation = comp;
        return o;
      }
      default:
        return { action_type: "classify", category: "general_inquiry", priority: "medium" };
    }
  }

  function syncJsonPreview() {
    $("actionJson").value = pretty(buildAction());
  }

  function showFieldGroups() {
    const t = $("actionType").value;
    document.querySelectorAll(".field-group").forEach((el) => {
      el.classList.toggle("active", el.id === "fg" + capitalize(t));
    });
    syncJsonPreview();
  }

  function capitalize(s) {
    return s.split("_").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join("");
  }

  ["actionType", "clsCategory", "clsPriority", "rteDept", "rspTone", "rspText",
   "escTeam", "escReason", "reqQ", "rsvSummary", "rsvComp"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("input", syncJsonPreview);
    if (el) el.addEventListener("change", syncJsonPreview);
  });
  $("actionType").addEventListener("change", showFieldGroups);

  function actionSummary(actionTaken) {
    if (!actionTaken) return "—";
    try {
      const o = JSON.parse(actionTaken);
      if (o && typeof o.action_type === "string") return o.action_type;
    } catch (e) { /* ignore */ }
    return actionTaken.length > 56 ? actionTaken.slice(0, 53) + "…" : actionTaken;
  }

  function renderTicket(obs, info) {
    const body = $("ticketBody");
    const pills = $("ticketPills");
    pills.innerHTML = "";
    if (!obs) {
      body.textContent = "No observation loaded.";
      body.classList.add("empty");
      return;
    }
    body.classList.remove("empty");
    body.textContent = obs.ticket_text != null ? obs.ticket_text : "";

    function addPill(label, value) {
      if (value === undefined || value === null || value === "") return;
      const p = document.createElement("div");
      p.className = "pill";
      const k = document.createElement("kbd");
      k.textContent = label + ":";
      const v = document.createElement("span");
      v.className = "val";
      v.textContent = String(value);
      p.appendChild(k);
      p.appendChild(v);
      pills.appendChild(p);
    }

    if (info && info.difficulty) addPill("Difficulty", String(info.difficulty));
    const impact = (obs.customer_value || "").toString().toUpperCase();
    addPill("Impact", impact);
    addPill("ID", obs.ticket_id);
    addPill("Tier", obs.customer_tier);
    addPill("Sentiment", obs.customer_sentiment);
    addPill("Phase", obs.phase);
    if (typeof obs.sla_steps_remaining === "number") {
      addPill("SLA steps left", String(obs.sla_steps_remaining));
    }
    if (obs.category_hint) addPill("Hint", obs.category_hint);
  }

  function renderTimeline(obs) {
    const list = $("timelineList");
    list.innerHTML = "";
    if (!obs || !Array.isArray(obs.history) || obs.history.length === 0) {
      const empty = document.createElement("div");
      empty.className = "timeline-empty";
      empty.innerHTML = "No actions yet. Run <strong>Reset</strong>, then <strong>Execute step</strong>.";
      list.appendChild(empty);
      return;
    }
    obs.history.forEach(function (h) {
      const item = document.createElement("div");
      item.className = "timeline-item";

      const row = document.createElement("div");
      row.className = "timeline-row";

      const main = document.createElement("div");
      main.className = "tl-main";
      const sn = document.createElement("span");
      sn.className = "tl-step-num";
      sn.textContent = "Step " + h.step + ": ";
      const an = document.createElement("span");
      an.className = "tl-action-name";
      an.textContent = actionSummary(h.action_taken);
      main.appendChild(sn);
      main.appendChild(an);

      const rw = document.createElement("div");
      rw.className = "tl-reward";
      const r = Number(h.reward_earned);
      rw.textContent = (r >= 0 ? "+" : "") + r.toFixed(2);
      if (r < 0) rw.classList.add("neg");
      else if (r > 0) rw.classList.add("pos");

      row.appendChild(main);
      row.appendChild(rw);
      item.appendChild(row);

      if (h.env_feedback) {
        const fb = document.createElement("div");
        fb.className = "tl-feedback";
        fb.textContent = h.env_feedback;
        item.appendChild(fb);
      }

      list.appendChild(item);
    });
  }

  function showPayload(data, opts) {
    const obs = data.observation;
    outObs.textContent = obs == null ? "null" : pretty(obs);
    outReward.textContent = String(data.reward);
    outDone.textContent = String(data.done);
    outInfo.textContent = pretty(data.info != null ? data.info : {});

    renderTicket(obs, data.info);
    renderTimeline(obs);

    mLast.textContent = Number(data.reward).toFixed(2);
    if (opts && opts.addStepReward && typeof data.reward === "number") {
      totalReward += data.reward;
    }
    mTotal.textContent = totalReward.toFixed(2);
    const ns = data.info && typeof data.info.normalized_score === "number"
      ? data.info.normalized_score
      : null;
    mScore.textContent = ns == null ? "—" : Math.round(ns * 100) + "%";
    mStat.textContent = data.done ? "DONE" : "RUNNING";
  }

  function setStatus(msg, isErr) {
    status.textContent = msg || "";
    status.className = isErr ? "err" : "";
  }

  async function parseJsonResponse(res) {
    const text = await res.text();
    try {
      return { ok: res.ok, data: JSON.parse(text), raw: text };
    } catch {
      return { ok: res.ok, data: null, raw: text };
    }
  }

  async function doReset() {
    setStatus("POST /reset …");
    const seedRaw = parseInt($("resetSeed").value, 10);
    const seed = Number.isFinite(seedRaw) ? seedRaw : 0;
    const diff = $("resetDifficulty").value;
    const payload = { seed: seed };
    if (diff) payload.difficulty = diff;
    const res = await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      setStatus((data && data.detail) || raw || res.statusText, true);
      return;
    }
    totalReward = 0;
    showPayload(data, {});
    mLast.textContent = "0.00";
    mTotal.textContent = "0.00";
    setStatus("POST /reset → " + res.status);
  }

  async function doState() {
    setStatus("GET /state …");
    const res = await fetch("/state", { method: "GET" });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      setStatus((data && data.detail) || raw || res.statusText, true);
      return;
    }
    showPayload(data, {});
    setStatus("GET /state → " + res.status);
  }

  async function doStep() {
    const action = buildAction();
    syncJsonPreview();
    setStatus("POST /step …");
    const res = await fetch("/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action }),
    });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      const detail = data && (data.detail || data.message);
      setStatus(String(detail || raw || res.statusText), true);
      return;
    }
    showPayload(data, { addStepReward: true });
    setStatus("POST /step → " + res.status);
  }

  $("btnReset").addEventListener("click", () => { doReset().catch((e) => setStatus(String(e), true)); });
  $("btnState").addEventListener("click", () => { doState().catch((e) => setStatus(String(e), true)); });
  $("btnStep").addEventListener("click", () => { doStep().catch((e) => setStatus(String(e), true)); });

  showFieldGroups();
})();
  </script>
</body>
</html>
"""


# ── routes ───────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(content=_DEBUG_UI_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=EnvResponse)
async def reset(req: ResetRequest | None = None) -> dict[str, Any]:
    if req is None:
        req = ResetRequest()
    try:
        result = await _env.reset(seed=req.seed, difficulty=req.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_to_dict(result)


@app.post("/step", response_model=EnvResponse)
async def step(req: StepRequest) -> dict[str, Any]:
    try:
        result = await _env.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _result_to_dict(result)


@app.get("/state")
async def state() -> dict[str, Any] | None:
    result = await _env.state()
    if result is None:
        return {"observation": None, "reward": 0.0, "done": False, "info": {}}
    return _result_to_dict(result)


@app.post("/close")
async def close() -> dict[str, str]:
    await _env.close()
    return {"status": "closed"}


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint() -> dict[str, Any]:
    import inference

    buf = io.StringIO()
    with redirect_stdout(buf):
        await inference.run()

    stdout = buf.getvalue()
    score = 0.0
    success = False
    for line in stdout.strip().splitlines():
        if line.startswith("[END]"):
            for part in line.split():
                if part.startswith("score="):
                    score = float(part.split("=")[1])
                if part.startswith("success="):
                    success = part.split("=")[1] == "true"

    return {"stdout": stdout, "score": score, "success": success}


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
