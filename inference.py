"""
PortOps-LLM — LLM Agent Inference Script
==========================================
Baseline script for running an OpenAI-compatible LLM agent
against the PortOps-LLM OpenEnv environment.

Usage:
    python inference.py --task 1 --seed 42 --model gpt-4o
    python inference.py --task 2 --seed 7  --base-url http://localhost:7860
    python inference.py --task 3 --seed 42 --verbose

Environment variables:
    OPENAI_API_KEY       — Required for OpenAI / compatible endpoints
    PORTOPS_BASE_URL     — Override server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import textwrap
from typing import Any, Dict, List, Optional

# Load .env file automatically (if present) before reading any env vars
from dotenv import load_dotenv
load_dotenv()

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("PORTOPS_MODEL", "gpt-4o-mini"))
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

DEFAULT_BASE_URL: str = os.getenv("PORTOPS_BASE_URL", "http://localhost:7860")
MAX_STEPS: int = 8
MAX_RETRIES_ON_PARSE_FAIL: int = 2   # extra LLM calls if action format is wrong


# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — compact version (~150 tokens, saves ~75% vs original)
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
Critical Instruction: You are a Port Controller. You have failed because you attempted to retrieve containers that were not requested. Follow these priority rules exactly:

COMMANDS:
  move(CONTAINER_ID, BAY_NUMBER)  — move TOP container of its bay to target bay (1–5)
  retrieve(CONTAINER_ID)          — ship out a container (must be top of its bay)

Output ONLY one command per turn — no explanation.

1. Priority 1: Outbound Requests
Check outbound_requests first.
If a container is listed there, move blockers and retrieve(ID).
If outbound_requests is empty/null, STOP trying to retrieve.

2. Priority 2: Inbound Queue (The Fix)
If outbound_requests is empty, check inbound_queue.
You must move containers from the queue into the yard using the format: move(INBOUND_ID, BAY_NUMBER).
Example: If inbound_queue is ["C13", "C14"], look at your ✅ SAFE MOVES and pick a valid move for C13.

3. Priority 3: Safety Check
Never issue a command that is not explicitly listed in the ✅ SAFE MOVES provided in the latest observation.
If you see ⚠️ Server error: Not requested, it means you are trying to retrieve a container that the client doesn't want. Switch to Priority 2 (Inbound Queue).

4. Logic Loop:
Is there an outbound request?
Yes -> Retrieve it (move blockers first if needed).
No -> Check Inbound Queue.
Is there an inbound container?
Yes -> Move it to a safe Bay listed in SAFE MOVES.
No -> If both lists are empty, you have finished the task.
CRITICAL RULES:
1. You can ONLY retrieve a container if its ID is explicitly listed in "outbound_requests" AND it is at the very top of its bay.
2. If "outbound_requests" is empty, your new goal is to place containers from the "inbound_queue" into the yard.
3. To place an inbound container, use: move(inbound_container_id, target_bay_number).
4. NEVER retrieve a container that is not in the outbound_requests list. 
5. Always pick an action from the "SAFE MOVES" list provided in your observation.
""").strip()


# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT CLIENT (wraps the HTTP API)
# ─────────────────────────────────────────────────────────────────

class PortOpsClient:
    """Thin HTTP client wrapper for the PortOps-LLM OpenEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def reset(self, task_id: int = 1, seed: int = 42) -> Dict[str, Any]:
        """Reset the environment and return the initial observation dict."""
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, command: str) -> Dict[str, Any]:
        """Execute a command and return the full step response dict."""
        resp = self.session.post(
            f"{self.base_url}/step",
            json={"command": command},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Return the raw internal state dict."""
        resp = self.session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────
# LLM INTERACTION
# ─────────────────────────────────────────────────────────────────

def build_user_message(obs: Dict[str, Any], step_num: int) -> str:
    """Build a user-turn prompt string from the observation dict."""
    yard_text = obs.get("yard_text", "(no yard data)")
    inbound = obs.get("inbound_queue", [])
    outbound = obs.get("outbound_requests", [])
    error = obs.get("last_action_error")
    remaining = obs.get("steps_remaining", MAX_STEPS - step_num)

    lines = [
        f"─── STEP {step_num} / {MAX_STEPS} ({remaining} remaining) ───",
        "",
        "CURRENT YARD STATE:",
        yard_text,
        "",
        f"INBOUND QUEUE  : {inbound if inbound else '(none)'}",
        f"OUTBOUND TARGET: {outbound if outbound else '(none — all retrieved!)'}",
    ]
    if error:
        lines.append(f"\n⚠️  YOUR LAST COMMAND WAS INVALID: {error}")
        lines.append("Please correct your command and try again.")

    lines.append("\nWhat is your next command? (Respond with ONLY the command, nothing else)")
    return "\n".join(lines)


def extract_command(llm_response: str) -> str:
    """
    Extract the command from the LLM's response string.
    Handles cases where the model adds surrounding text despite instructions.
    """
    text = llm_response.strip()

    # Try direct match first
    patterns = [
        r"move\s*\(\s*[A-Za-z0-9]+\s*,\s*[1-5]\s*\)",
        r"retrieve\s*\(\s*[A-Za-z0-9]+\s*\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    # Fallback: return first line if it looks like a command
    first_line = text.split("\n")[0].strip()
    return first_line


def call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    verbose: bool = False,
) -> str:
    """
    Call the LLM and return its text response.
    Gemini-compatible: no stop sequences (Gemini ignores/errors on them),
    reasoning_effort='none' disables thinking tokens on Gemini 2.5 models
    (saves ~50% cost). Falls back gracefully for non-Gemini models.
    """
    is_gemini = "gemini" in model.lower()

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0.0,    # Deterministic for evaluation
        max_tokens=128,     # Generous — commands are short but Gemini needs headroom
    )

    # Gemini 2.5 models support reasoning_effort to disable thinking tokens
    if is_gemini:
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    else:
        # For OpenAI/other providers, stop at newline — commands are single-line
        kwargs["stop"] = ["\n"]

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    # Always strip and take only the first line (guards against Gemini verbosity)
    first_line = content.strip().split("\n")[0].strip()

    if verbose:
        print(f"  [LLM] raw response : {repr(content)}")
        print(f"  [LLM] first line   : {repr(first_line)}")
    return first_line


# ─────────────────────────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────────

def run_agent(
    task_id: int,
    seed: int,
    model: str,
    base_url: str,
    openai_api_key: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run the full agent episode for the given task.
    Returns a results dict containing the final score and trajectory.
    """
    task_names = {1: "The Extraction", 2: "Temporal Allocation", 3: "Hazmat & Weight"}
    print(f"\n{'═' * 65}")
    print(f"  PortOps-LLM — Task {task_id}: {task_names.get(task_id, '?')}")
    print(f"  Model: {model} | Seed: {seed} | Server: {base_url}")
    print(f"{'═' * 65}\n")

    # ── Connect to environment ─────────────────────────────────
    env = PortOpsClient(base_url)
    if not env.health_check():
        print(f"❌ ERROR: Cannot reach PortOps server at {base_url}")
        print("   Start the server first: uvicorn server.app:app --port 7860")
        sys.exit(1)

    client_api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or HF_TOKEN or "dummy-key-for-local"
    
    llm = OpenAI(
        api_key=client_api_key,
        base_url=API_BASE_URL,
    )

    # ── Reset environment ──────────────────────────────────────
    print(f"[START] Resetting environment for Task {task_id} (seed={seed})…")
    obs = env.reset(task_id=task_id, seed=seed)
    if verbose:
        print(json.dumps(obs, indent=2))

    trajectory: List[Dict[str, Any]] = []
    final_score: float = 0.0
    done: bool = False

    # ── Agent loop ─────────────────────────────────────────────
    # No history accumulation — each call is [system + current_obs] only.
    # The observation is self-contained (full yard state every step).
    for step_num in range(1, MAX_STEPS + 1):

        # Build user message from current observation
        user_msg = build_user_message(obs, step_num)

        if verbose:
            print(f"\n{'─' * 45}")
            print(f"STEP {step_num} OBSERVATION:")
            print(obs.get("yard_text", ""))

        # ── LLM call with retry on bad format ─────────────────
        command = ""
        for attempt in range(MAX_RETRIES_ON_PARSE_FAIL + 1):
            # Fresh [system + user] each call — no accumulating history
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
            raw = call_llm(llm, model, messages, verbose)
            command = extract_command(raw)

            # Validate format locally before sending
            import re as _re
            valid = bool(
                _re.fullmatch(r"move\s*\(\s*[A-Za-z0-9]+\s*,\s*[1-5]\s*\)", command, _re.I)
                or _re.fullmatch(r"retrieve\s*\(\s*[A-Za-z0-9]+\s*\)", command, _re.I)
            )
            if valid:
                break
            # On bad format, append correction to the same mini-conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"INVALID: '{raw}'. Respond with ONLY:\n"
                    "  move(ID, BAY)  or  retrieve(ID)\nTry again:"
                )
            })

        print(f"[STEP {step_num:02d}] Command: {command}")

        # ── Execute step ───────────────────────────────────────
        try:
            result = env.step(command)
        except requests.HTTPError as e:
            print(f"  ⚠️  HTTP Error: {e}")
            trajectory.append({"step": step_num, "command": command, "error": str(e)})
            break

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        trajectory.append({
            "step": step_num,
            "command": command,
            "reward": reward,
            "done": done,
            "info": info,
            "error": obs.get("last_action_error"),
        })

        if verbose:
            print(f"  reward={reward:.4f}, done={done}")
            if obs.get("last_action_error"):
                print(f"  ⚠️  Server error: {obs['last_action_error']}")

        # ── Check terminal conditions ──────────────────────────
        if done:
            final_score = info.get("score", reward)
            reason = info.get("reason", "unknown")
            print(f"\n[END] Episode finished — reason: {reason}")
            break

    # ── Final report ───────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  FINAL SCORE : {final_score:.4f} / 1.0000")
    print(f"  STEPS TAKEN : {step_num}")
    print(f"  TASK        : {task_id} — {task_names.get(task_id, '?')}")
    print(f"{'═' * 65}\n")

    results = {
        "task_id": task_id,
        "seed": seed,
        "model": model,
        "final_score": final_score,
        "steps_taken": step_num,
        "done": done,
        "trajectory": trajectory,
    }

    # Optionally save results to JSON
    results_path = f"results_task{task_id}_seed{seed}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PortOps-LLM Inference Script — Run an LLM agent against the CRP environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python inference.py --task 1 --seed 42 --model gpt-4o
              python inference.py --task 2 --seed 7 --verbose
              python inference.py --task 3 --base-url http://localhost:7860
        """),
    )
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3],
                        help="Task ID: 1=Extraction, 2=Temporal, 3=Hazmat (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic environment setup (default: 42)")
    parser.add_argument("--model", type=str,
                        default=MODEL_NAME,
                        help=f"OpenAI model name (default: {MODEL_NAME})")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help=f"PortOps server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed step information and LLM responses")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(  
        task_id=args.task,
        seed=args.seed,
        model=args.model,
        base_url=args.base_url,
        openai_api_key=args.api_key,
        verbose=args.verbose,
    )
