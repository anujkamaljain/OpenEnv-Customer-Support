"""FastAPI server exposing the CustomerSupportEnv over HTTP.

Endpoints:
    POST /reset         — start a new episode
    POST /step          — apply an action
    GET  /state         — read current state without advancing
    POST /close         — release episode resources
    POST /inference     — run the full LLM inference loop
    GET  /health        — liveness probe
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any

from fastapi import FastAPI, HTTPException
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


# ── routes ───────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=EnvResponse)
async def reset(req: ResetRequest | None = None) -> dict[str, Any]:
    seed = req.seed if req else 0
    difficulty = req.difficulty if req else None
    result = await _env.reset(seed=seed, difficulty=difficulty)
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
            break

    return {"stdout": stdout, "score": score, "success": success}


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
