"""FastAPI server exposing the CustomerSupportEnv over HTTP.

Endpoints:
    GET  /              — browser debug UI (HTML)
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
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import CustomerSupportEnv

app = FastAPI(title="OpenEnv Customer Support", version="0.1.0")

_env = CustomerSupportEnv()


# ── request / response schemas ───────────────────────────────────────────────


class ResetRequest(BaseModel):
    seed: int = 0
    difficulty: str | None = None


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
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .wrap { max-width: 56rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
    .title-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem; }
    .title-row span { font-size: 1.75rem; }
    h1 { font-size: 1.35rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title-row">
      <span aria-hidden="true">🤖</span>
      <h1>Customer Support Command Center</h1>
    </div>
    <p class="sub">Classify → route → respond / escalate / request info → resolve ·
      <a href="/health" style="color:var(--accent)">/health</a></p>

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

    <div class="panel">
      <div class="hint-bar">
        Follow the env phase: start with <strong>classify</strong>, then <strong>route</strong>, then optional
        <strong>respond</strong> / <strong>escalate</strong> / <strong>request_info</strong>, then <strong>resolve</strong>.
        Use <strong>Get state</strong> to see <code>available_actions</code>.
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

      <div class="btn-row">
        <button type="button" class="exec" id="btnStep">Execute step</button>
        <button type="button" class="secondary" id="btnReset">Reset</button>
        <button type="button" class="secondary" id="btnState">Get state</button>
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

  function showPayload(data, opts) {
    outObs.textContent = data.observation == null ? "null" : pretty(data.observation);
    outReward.textContent = String(data.reward);
    outDone.textContent = String(data.done);
    outInfo.textContent = pretty(data.info != null ? data.info : {});

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
    const res = await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
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
    result = await _env.reset(seed=req.seed, difficulty=req.difficulty)
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
