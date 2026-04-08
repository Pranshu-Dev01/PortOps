"""
PortOps-LLM — FastAPI Server
=============================
OpenEnv-compliant HTTP server exposing the PortOpsEnv via:
  POST /reset  → initial ObservationSpace
  POST /step   → (ObservationSpace, reward, done, info)
  GET  /state  → raw state dict
  GET  /health → liveness probe

Run with:
  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root when launched from subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import (
    ActionSpace,
    ObservationSpace,
    PortOpsEnv,
)

# ─────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(
        default=1, ge=1, le=3,
        description="Task to run: 1=Extraction, 2=Temporal Allocation, 3=Hazmat/Weight"
    )
    seed: int = Field(
        default=42, description="Random seed for deterministic environment setup"
    )


class StepRequest(BaseModel):
    command: str = Field(
        ...,
        description=(
            "Action command string. "
            "Format: 'move(CONTAINER_ID, BAY_NUMBER)' or 'retrieve(CONTAINER_ID)'. "
            "Bay numbers are 1-based (1–5)."
        )
    )


class StepResponse(BaseModel):
    observation: ObservationSpace
    reward: float
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────

app = FastAPI(
    title="PortOps-LLM",
    description=(
        "Container Relocation Problem (CRP) environment for LLM agent evaluation. "
        "Three tasks of increasing difficulty on a 5-Bay × 4-Tier container yard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process (stateful)
_env = PortOpsEnv()


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health_check() -> Dict[str, str]:
    """Liveness probe — always returns 200 OK."""
    return {"status": "ok", "service": "PortOps-LLM"}


@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    """Root endpoint with environment metadata."""
    return {
        "name": "PortOps-LLM",
        "description": "Container Relocation Problem — OpenEnv LLM Evaluation Environment",
        "tasks": {
            "1": "The Extraction (Easy)         — Unstack and retrieve a buried container",
            "2": "Temporal Allocation (Medium)  — Sort inbound containers by departure day",
            "3": "Hazmat & Weight Constraints (Hard) — Dense yard with fatal violations",
        },
        "max_steps": 8,
        "yard_size": "5 Bays × 4 Tiers",
        "action_format": [
            "move(CONTAINER_ID, BAY_NUMBER)   # e.g. move(C03, 2)",
            "retrieve(CONTAINER_ID)            # e.g. retrieve(C01)",
        ],
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
    }


@app.post("/reset", response_model=ObservationSpace, tags=["environment"])
def reset(body: Optional[ResetRequest] = None) -> ObservationSpace:
    """
    Reset the environment to the start of a new episode.
    Returns the initial observation.
    """
    if body is None:
        body = ResetRequest()
    try:
        obs = _env.reset(task_id=body.task_id, seed=body.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["environment"])
def step(body: StepRequest) -> StepResponse:
    """
    Execute one action in the environment.
    Returns the updated observation, reward, done flag, and info dict.
    """
    action = ActionSpace(command=body.command)
    obs, reward, done, info = _env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", tags=["environment"])
def state() -> Dict[str, Any]:
    """
    Return the raw internal state of the environment.
    Useful for debugging and evaluation harnesses.
    """
    return _env.state()


# ─────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
