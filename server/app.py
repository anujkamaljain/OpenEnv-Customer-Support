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
  <title>OpenEnv — debug</title>
  <style>
    :root {
      --bg: #1a1b1e;
      --surface: #25262b;
      --border: #373a40;
      --text: #c1c2c5;
      --muted: #868e96;
      --accent: #4c6ef5;
      --accent-hover: #5c7cfa;
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
    .wrap {
      max-width: 52rem;
      margin: 0 auto;
      padding: 1.5rem 1rem 3rem;
    }
    h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.25rem; }
    .sub { color: var(--muted); font-size: 0.875rem; margin-bottom: 1.25rem; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.1rem;
      margin-bottom: 1rem;
    }
    .row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 0.75rem; }
    .row:last-child { margin-bottom: 0; }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 0.45rem 0.9rem;
      border-radius: 6px;
      font-size: 0.875rem;
      cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button.secondary { background: #495057; }
    button.secondary:hover { background: #5c636a; }
    label { font-size: 0.8rem; color: var(--muted); display: block; margin-bottom: 0.35rem; }
    textarea {
      width: 100%;
      min-height: 8rem;
      padding: 0.6rem 0.65rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #141517;
      color: var(--text);
      font-family: var(--mono);
      font-size: 0.8rem;
      resize: vertical;
    }
    pre {
      margin: 0;
      padding: 0.65rem 0.75rem;
      background: #141517;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 0.75rem;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .grid { display: grid; gap: 0.75rem; }
    @media (min-width: 640px) { .grid-2 { grid-template-columns: 1fr 1fr; } }
    .tag { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.25rem; }
    #status { font-size: 0.8rem; min-height: 1.25rem; }
    #status.err { color: var(--danger); }
    footer { margin-top: 1.5rem; font-size: 0.75rem; color: var(--muted); }
    footer a { color: var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>OpenEnv — customer support</h1>
    <p class="sub">Debug UI · same API as Postman · <a href="/health" style="color:var(--accent)">/health</a> (JSON)</p>

    <div class="panel">
      <div class="row">
        <button type="button" id="btnReset">Reset environment</button>
        <button type="button" id="btnState" class="secondary">Get state</button>
        <button type="button" id="btnStep">Step</button>
      </div>
      <label for="actionJson">Action JSON (sent as <code>POST /step</code> body <code>{"action": …}</code>)</label>
      <textarea id="actionJson" spellcheck="false">{
  "action_type": "classify",
  "category": "general_inquiry",
  "priority": "medium"
}</textarea>
    </div>

    <p id="status"></p>

    <div class="panel grid grid-2">
      <div>
        <div class="tag">reward</div>
        <pre id="outReward">—</pre>
      </div>
      <div>
        <div class="tag">done</div>
        <pre id="outDone">—</pre>
      </div>
    </div>
    <div class="panel">
      <div class="tag">observation</div>
      <pre id="outObs">—</pre>
    </div>
    <div class="panel">
      <div class="tag">info</div>
      <pre id="outInfo">—</pre>
    </div>

    <footer>Uses relative <code>fetch</code> URLs — works on localhost and Hugging Face Spaces.</footer>
  </div>
  <script>
(function () {
  const $ = (id) => document.getElementById(id);
  const status = $("status");
  const outObs = $("outObs");
  const outReward = $("outReward");
  const outDone = $("outDone");
  const outInfo = $("outInfo");

  function pretty(obj) {
    return JSON.stringify(obj, null, 2);
  }

  function showPayload(data) {
    outObs.textContent = data.observation == null ? "null" : pretty(data.observation);
    outReward.textContent = String(data.reward);
    outDone.textContent = String(data.done);
    outInfo.textContent = pretty(data.info != null ? data.info : {});
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
    showPayload(data);
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
    showPayload(data);
    setStatus("GET /state → " + res.status);
  }

  async function doStep() {
    let action;
    try {
      action = JSON.parse($("actionJson").value);
    } catch (e) {
      setStatus("Invalid JSON in action textarea: " + e.message, true);
      return;
    }
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
    showPayload(data);
    setStatus("POST /step → " + res.status);
  }

  $("btnReset").addEventListener("click", () => { doReset().catch((e) => setStatus(String(e), true)); });
  $("btnState").addEventListener("click", () => { doState().catch((e) => setStatus(String(e), true)); });
  $("btnStep").addEventListener("click", () => { doStep().catch((e) => setStatus(String(e), true)); });
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
