"""
PortOps-LLM inference runner with strict evaluator logging.

Mandatory environment variables expected in submission config:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
- LOCAL_IMAGE_NAME (declared for compatibility when docker image mode is used)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

PORTOPS_BASE_URL = os.getenv("PORTOPS_BASE_URL", "http://localhost:7860")
MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 96
SUCCESS_SCORE_THRESHOLD = 0.1

TASK_NAMES = {
    1: "the-extraction",
    2: "temporal-allocation",
    3: "hazmat-weight-constraints",
}
BENCHMARK = "portops-llm"

SYSTEM_PROMPT = (
    "Return exactly one action command each turn. "
    "Allowed formats: move(CONTAINER_ID, BAY_NUMBER) or retrieve(CONTAINER_ID). "
    "Use only legal actions from SAFE MOVES when present."
)


class PortOpsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def reset(self, task_id: int, seed: int) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def step(self, command: str) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/step",
            json={"command": command},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    done_value = str(done).lower()
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def _extract_safe_moves(yard_text: str) -> List[str]:
    match = re.search(
        r"SAFE MOVES:\s*(.+)$", yard_text, flags=re.IGNORECASE | re.MULTILINE
    )
    if not match:
        return []
    return [entry.strip() for entry in match.group(1).split(",") if entry.strip()]


def _is_valid_command(command: str) -> bool:
    move_ok = bool(
        re.fullmatch(
            r"move\(\s*[A-Za-z0-9]+\s*,\s*[1-5]\s*\)", command, re.IGNORECASE
        )
    )
    retrieve_ok = bool(
        re.fullmatch(r"retrieve\(\s*[A-Za-z0-9]+\s*\)", command, re.IGNORECASE)
    )
    return move_ok or retrieve_ok


def _extract_command(text: str) -> str:
    if not text:
        return ""
    move = re.search(
        r"move\(\s*[A-Za-z0-9]+\s*,\s*[1-5]\s*\)", text, flags=re.IGNORECASE
    )
    if move:
        return move.group(0).replace(" ", "")
    retrieve = re.search(
        r"retrieve\(\s*[A-Za-z0-9]+\s*\)", text, flags=re.IGNORECASE
    )
    if retrieve:
        return retrieve.group(0).replace(" ", "")
    first_line = text.strip().splitlines()[0].strip()
    return first_line


def _fallback_action(observation: Dict[str, Any]) -> str:
    safe_moves = _extract_safe_moves(observation.get("yard_text", ""))
    if safe_moves:
        return safe_moves[0]

    outbound = observation.get("outbound_requests") or []
    if outbound:
        return f"retrieve({outbound[0]})"

    inbound = observation.get("inbound_queue") or []
    if inbound:
        return f"move({inbound[0]},1)"

    return "retrieve(C01)"


def _build_user_prompt(observation: Dict[str, Any], step: int) -> str:
    return (
        f"Step: {step}\n"
        f"Yard:\n{observation.get('yard_text', '')}\n"
        f"Inbound: {observation.get('inbound_queue', [])}\n"
        f"Outbound: {observation.get('outbound_requests', [])}\n"
        f"Last error: {observation.get('last_action_error')}\n"
        "Return exactly one action command."
    )


def _next_action(
    llm: OpenAI,
    observation: Dict[str, Any],
    step: int,
) -> str:
    fallback = _fallback_action(observation)
    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        candidate = _extract_command(raw)
        if _is_valid_command(candidate):
            return candidate
    except Exception:
        pass
    return fallback


def run_episode(task_id: int, seed: int, base_url: str, api_key: str) -> None:
    env = PortOpsClient(base_url)
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")

    rewards: List[float] = []
    score = 0.0
    success = False
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not env.health_check():
            raise RuntimeError("environment_unreachable")

        observation = env.reset(task_id=task_id, seed=seed)
        llm = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        for step in range(1, MAX_STEPS + 1):
            action = _next_action(llm, observation, step)
            result = env.step(action)

            observation = result.get("observation", {})
            reward = float(result.get("reward") or 0.0)
            done = bool(result.get("done", False))
            error = observation.get("last_action_error")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                info = result.get("info", {})
                score = float(info.get("score", reward))
                break

        if score == 0.0 and rewards:
            score = float(rewards[-1])

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        success = False
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PortOps inference episode")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default=PORTOPS_BASE_URL)
    parser.add_argument("--api-key", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    key = args.api_key or HF_TOKEN or os.getenv("OPENAI_API_KEY") or "dummy-key"
    run_episode(
        task_id=args.task, seed=args.seed, base_url=args.base_url, api_key=key
    )
