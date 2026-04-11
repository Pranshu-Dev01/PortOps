#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@"
  fi
}

ARG1="${1:-}"
ARG2="${2:-}"
REPO_DIR="."
PING_URL=""

if [ -n "$ARG1" ] && [[ "$ARG1" =~ ^https?:// ]]; then
  PING_URL="$ARG1"
  REPO_DIR="${ARG2:-.}"
elif [ -n "$ARG1" ]; then
  REPO_DIR="$ARG1"
  PING_URL="${HF_SPACE_URL:-}"
else
  REPO_DIR="${ARG2:-.}"
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "Repo path not found: ${2:-.}"
  exit 1
fi

# Load local .env so validator checks match repo configuration.
if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_DIR/.env"
  set +a
fi

if [ -z "$PING_URL" ]; then
  PING_URL="${HF_SPACE_URL:-}"
fi

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 [ping_url] [repo_dir]"
  echo "Provide ping_url directly or set HF_SPACE_URL in your shell/.env."
  echo "Example: HF_SPACE_URL=https://my-space.hf.space $0 ."
  exit 1
fi

PING_URL="${PING_URL%/}"

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
warn() { log "${YELLOW}WARN${NC} -- $1"; }

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/5: Pinging HF Space reset endpoint${NC}"
HTTP_CODE=$(curl -s -o /tmp/openenv_ping_resp.$$ -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || printf "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space responds to /reset"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
  exit 1
fi

log "${BOLD}Step 2/5: Docker build${NC}"
if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  exit 1
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server directory"
  exit 1
fi

BUILD_OK=false
run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" >/tmp/openenv_docker_build.$$ 2>&1 && BUILD_OK=true

if [ "$BUILD_OK" != true ] && grep -Eiq "unauthorized|token is expired|access token" /tmp/openenv_docker_build.$$; then
  warn "Docker auth failed; retrying build with anonymous Docker config"
  CLEAN_DOCKER_CONFIG="/tmp/openenv-docker-config-$$"
  mkdir -p "$CLEAN_DOCKER_CONFIG"
  BUILD_OK=false
  run_with_timeout "$DOCKER_BUILD_TIMEOUT" env DOCKER_CONFIG="$CLEAN_DOCKER_CONFIG" docker build "$DOCKER_CONTEXT" >/tmp/openenv_docker_build.$$ 2>&1 && BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  tail -20 /tmp/openenv_docker_build.$$ || true
  exit 1
fi

log "${BOLD}Step 3/5: OpenEnv validate${NC}"
if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  exit 1
fi

if (cd "$REPO_DIR" && openenv validate --verbose >/tmp/openenv_validate.$$ 2>&1); then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  cat /tmp/openenv_validate.$$ || true
  exit 1
fi

log "${BOLD}Step 4/5: Baseline inference run${NC}"
if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "inference.py not found in repo root"
  exit 1
fi

if [ -z "${API_BASE_URL:-}" ]; then
  warn "API_BASE_URL is not set in environment"
fi
if [ -z "${MODEL_NAME:-}" ]; then
  warn "MODEL_NAME is not set in environment"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  warn "HF_TOKEN is not set in environment"
fi

(
  cd "$REPO_DIR" || exit 1
  source .venv/bin/activate 2>/dev/null || true

  run_with_timeout 1200 python inference.py --task 1 --seed 42 --base-url "$PING_URL" >/tmp/openenv_inference.$$ 2>&1
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "inference_nonzero"
    exit 1
  fi

  START_COUNT=$(grep -c '^\[START\]' /tmp/openenv_inference.$$ || true)
  STEP_COUNT=$(grep -c '^\[STEP\]' /tmp/openenv_inference.$$ || true)
  END_COUNT=$(grep -c '^\[END\]' /tmp/openenv_inference.$$ || true)

  if [ "$START_COUNT" -lt 1 ] || [ "$END_COUNT" -ne 1 ] || [ "$STEP_COUNT" -lt 1 ]; then
    echo "invalid_inference_logs"
    cat /tmp/openenv_inference.$$ || true
    exit 1
  fi
)

if [ $? -eq 0 ]; then
  pass "inference.py completed with START/STEP/END output"
else
  fail "inference baseline check failed"
  exit 1
fi

log "${BOLD}Step 5/5: Task graders score range check${NC}"
(
  cd "$REPO_DIR" || exit 1
  source .venv/bin/activate 2>/dev/null || true
  python - <<'PY'
from env import PortOpsEnv, ActionSpace, MAX_STEPS

def run_task1():
    env = PortOpsEnv()
    env.reset(task_id=1, seed=42)
    env.step(ActionSpace(command="move(C03, 2)"))
    env.step(ActionSpace(command="move(C02, 3)"))
    _, _, done, info = env.step(ActionSpace(command="retrieve(C01)"))
    assert done
    return info["score"]

def run_task2():
    env = PortOpsEnv()
    obs = env.reset(task_id=2, seed=42)
    done = False
    info = {"score": 0.0}
    for _ in range(MAX_STEPS):
        if done or not obs.inbound_queue:
            break
        cmd = f"move({obs.inbound_queue[0]}, 1)"
        obs, _, done, info = env.step(ActionSpace(command=cmd))
    assert done
    return info["score"]

def run_task3():
    env = PortOpsEnv()
    env.reset(task_id=3, seed=42)
    _, _, done, info = env.step(ActionSpace(command="move(C13, 1)"))
    assert done
    return info["score"]

scores = {
    "task1": run_task1(),
    "task2": run_task2(),
    "task3": run_task3(),
}

for task, score in scores.items():
  if not (0.0 < float(score) < 1.0):
        raise SystemExit(f"{task} score out of range: {score}")

print(scores)
PY
)

if [ $? -eq 0 ]; then
  pass "All task graders returned scores in (0.0, 1.0)"
else
  fail "Task grader score-range check failed"
  exit 1
fi

printf "\n${GREEN}${BOLD}All checks passed. Submission looks ready.${NC}\n\n"
exit 0
