# PortOps-LLM 🚢

**Container Relocation Problem (CRP) — OpenEnv LLM Evaluation Environment**

An international hackathon environment where an LLM agent acts as a Terminal Operating System (TOS) managing a 5-Bay × 4-Tier container yard.

---

## Overview

The agent receives a **text-based observation** of the yard and must issue **text-based commands** (`move` or `retrieve`) to efficiently sort and retrieve containers while respecting physical constraints.

- **Yard**: 5 Bays wide, up to 4 Tiers high
- **Episode limit**: Strictly 8 steps
- **Action format**: `move(CONTAINER_ID, BAY_NUMBER)` or `retrieve(CONTAINER_ID)`

---

## Tasks

| # | Name | Difficulty | Description |
|---|------|------------|-------------|
| 1 | The Extraction | 🟢 Easy | Unstack and retrieve a buried container (no weight/hazmat constraints) |
| 2 | Temporal Allocation | 🟡 Medium | Place 8 inbound containers sorted by departure day |
| 3 | Hazmat & Weight Constraints | 🔴 Hard | Dense yard — fatal violations on Heavy-on-Light stacking or Hazmat adjacency |

---

## Project Structure

```
PortOps/
├── env.py              # Core environment: Pydantic models, physics engine, 3 tasks + graders
├── server/
│   ├── __init__.py
│   └── app.py          # FastAPI server exposing /reset, /step, /state endpoints
├── inference.py        # LLM agent baseline script (OpenAI API)
├── openenv.yaml        # OpenEnv-compliant manifest
├── Dockerfile          # Lightweight Docker image for HF Spaces (port 7860)
├── requirements.txt    # Python dependencies
├── pyproject.toml      # PEP 517 build config (required by openenv validate)
└── tests/
    └── test_env.py     # Unit tests for all 3 tasks
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

The interactive API docs will be available at: http://localhost:7860/docs

### 3. Run the baseline inference agent

```bash
export OPENAI_API_KEY="sk-..."
python inference.py --task 1 --seed 42 --model gpt-4o-mini --verbose
```

### 4. Validate OpenEnv compliance

```bash
openenv validate --verbose
```

---

## API Reference

### `POST /reset`
Reset the environment for a task.

```json
{ "task_id": 1, "seed": 42 }
```

### `POST /step`
Execute one action.

```json
{ "command": "move(C03, 2)" }
```
or
```json
{ "command": "retrieve(C01)" }
```

### `GET /state`
Returns the raw internal state (JSON).

---

## Action Format

| Action | Format | Example |
|--------|--------|---------|
| Move container | `move(ID, BAY)` | `move(C03, 2)` |
| Retrieve container | `retrieve(ID)` | `retrieve(C01)` |

- `ID` = container identifier (e.g. `C01`, `C12`)
- `BAY` = integer 1–5 (1-based)
- Only the **top container** of a bay can be moved or retrieved

---

## Scoring

| Task | Formula |
|------|---------|
| Task 1 | `max(0.0, 1.0 - 0.2 × (actual_moves - optimal_moves))` |
| Task 2 | `max(0.0, 1.0 - 0.15 × temporal_inversions)` |
| Task 3 | `max(0.0, 1.0 - 0.1 × unnecessary_steps)` — or **0.0** on fatal error |

### Task 3 Fatal Errors (score = 0.0 immediately)
- ⛔ Placing a **Heavy** container on top of a **Light** container
- ⛔ Placing a **Hazmat** container in a bay **adjacent** to another hazmat bay

---

## Docker / Hugging Face Spaces

```bash
docker build -t portops-llm .
docker run -p 7860:7860 portops-llm
```

For Hugging Face Spaces, the `Dockerfile` is configured to use `sdk: docker` on port `7860`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
