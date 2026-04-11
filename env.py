from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_BAYS: int = 5  # width of the yard
MAX_TIERS: int = 4  # maximum stack height per bay
MAX_STEPS: int = 8  # hard episode limit across all tasks
MIN_SCORE: float = 0.01
MAX_SCORE: float = 0.99


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────


class Container(BaseModel):
    """Represents a single shipping container."""

    id: str = Field(..., description="Unique container identifier, e.g. 'C01'")
    weight: Literal["Heavy", "Light"] = Field(
        default="Light", description="Weight class of the container"
    )
    is_hazmat: bool = Field(
        default=False, description="Whether the container holds hazardous material"
    )
    departure_day: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Scheduled departure day (1 = earliest, 5 = latest)",
    )

    def __repr__(self) -> str:
        flags = []
        if self.weight == "Heavy":
            flags.append("H")
        if self.is_hazmat:
            flags.append("HZ")
        tag = f"({','.join(flags)})" if flags else ""
        return f"{self.id}{tag}"


class YardState(BaseModel):
    """
    The 5x4 container yard grid.
    bays: List[List[Container]] — outer list = 5 bays (index 0–4),
          inner list = containers stacked bottom→top (index 0 = ground level).
    """

    bays: List[List[Container]] = Field(
        default_factory=lambda: [[] for _ in range(NUM_BAYS)],
        description="5 bays, each holding up to 4 containers (bottom-to-top order)",
    )

    @model_validator(mode="after")
    def validate_bay_count(self) -> "YardState":
        if len(self.bays) != NUM_BAYS:
            raise ValueError(f"YardState must have exactly {NUM_BAYS} bays")
        for i, bay in enumerate(self.bays):
            if len(bay) > MAX_TIERS:
                raise ValueError(
                    f"Bay {i} exceeds maximum tier height of {MAX_TIERS}"
                )
        return self

    def find_container(self, container_id: str) -> Optional[Tuple[int, int]]:
        """Return (bay_index, tier_index) of a container, or None if not found."""
        for bay_idx, bay in enumerate(self.bays):
            for tier_idx, c in enumerate(bay):
                if c.id == container_id:
                    return (bay_idx, tier_idx)
        return None

    def is_accessible(self, container_id: str) -> bool:
        """True iff the container is at the TOP of its bay (tier == len(bay)-1)."""
        loc = self.find_container(container_id)
        if loc is None:
            return False
        bay_idx, tier_idx = loc
        return tier_idx == len(self.bays[bay_idx]) - 1

    def top_container(self, bay_idx: int) -> Optional[Container]:
        """Return the container at the top of a bay, or None if empty."""
        bay = self.bays[bay_idx]
        return bay[-1] if bay else None

    def all_container_ids(self) -> List[str]:
        """Return a flat list of all container IDs currently in the yard."""
        return [c.id for bay in self.bays for c in bay]

    def render_text(self) -> str:
        """Render the yard as a human-readable text grid."""
        lines = ["┌─── CONTAINER YARD (5 Bays × 4 Tiers) ───────────────────────┐"]
        for tier in range(MAX_TIERS - 1, -1, -1):
            row_parts = []
            for bay in self.bays:
                if tier < len(bay):
                    row_parts.append(f"{repr(bay[tier]):^14}")
                else:
                    row_parts.append(f"{'[  empty  ]':^14}")
            tier_label = f"Tier {tier + 1}"
            lines.append(f"│ {tier_label} │ {'│'.join(row_parts)} │")
        lines.append(
            "├─────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
        )
        bay_labels = (
            "│         │"
            + "│".join(f"{'Bay ' + str(i + 1):^14}" for i in range(NUM_BAYS))
            + "│"
        )
        lines.append(bay_labels)
        lines.append(
            "└─────────────────────────────────────────────────────────────────────────┘"
        )
        return "\n".join(lines)

    def bay_summary(self) -> str:
        """Compact one-line-per-bay summary string."""
        parts = []
        for i, bay in enumerate(self.bays):
            if bay:
                contents = ", ".join(repr(c) for c in bay)
                parts.append(
                    f"Bay {i + 1} [{len(bay)}/{MAX_TIERS}]: {contents}  (bottom→top)"
                )
            else:
                parts.append(f"Bay {i + 1} [0/{MAX_TIERS}]: (empty)")
        return "\n".join(parts)


# ─────────────────────────────────────────────
# OBSERVATION & ACTION SPACE
# ─────────────────────────────────────────────


class ObservationSpace(BaseModel):
    yard_text: str
    inbound_queue: List[str]
    outbound_requests: List[str]
    last_action_error: Optional[str] = None
    step_count: int = 0
    steps_remaining: int = MAX_STEPS

    def to_prompt_str(self) -> str:
        lines = [
            "═" * 65,
            "  PORT OPERATIONS — YARD OBSERVATION",
            "═" * 65,
            "",
            self.yard_text,
            "",
            f"📦 INBOUND QUEUE  : {self.inbound_queue if self.inbound_queue else '(none)'}",
            f"🚢 OUTBOUND TARGET: {self.outbound_requests if self.outbound_requests else '(none)'}",
            f"⏱  STEP           : {self.step_count} / {MAX_STEPS} "
            f"({self.steps_remaining} remaining)",
        ]
        if self.last_action_error:
            lines.append(f"⚠️  LAST ERROR     : {self.last_action_error}")
        lines.append("═" * 65)
        return "\n".join(lines)


class ActionSpace(BaseModel):
    command: str = Field(..., min_length=1)

    def parse(self) -> Tuple[str, List[str]]:
        cmd = self.command.strip()
        move_match = re.fullmatch(
            r"move\(\s*([A-Za-z0-9]+)\s*,\s*([1-5])\s*\)", cmd, re.IGNORECASE
        )
        if move_match:
            return ("move", [move_match.group(1).upper(), move_match.group(2)])

        retrieve_match = re.fullmatch(
            r"retrieve\(\s*([A-Za-z0-9]+)\s*\)", cmd, re.IGNORECASE
        )
        if retrieve_match:
            return ("retrieve", [retrieve_match.group(1).upper()])

        raise ValueError(
            f"Invalid command format: '{cmd}'. Use move(ID, BAY) or retrieve(ID)."
        )


# ─────────────────────────────────────────────
# PHYSICS ENGINE HELPERS
# ─────────────────────────────────────────────


def _compute_min_moves_extraction(yard: YardState, target_id: str) -> int:
    loc = yard.find_container(target_id)
    if loc is None:
        return 0
    bay_idx, tier_idx = loc
    blocking = len(yard.bays[bay_idx]) - 1 - tier_idx
    return blocking + 1


def _count_temporal_inversions(yard: YardState) -> int:
    inversions = 0
    for bay in yard.bays:
        for i in range(len(bay) - 1):
            if bay[i + 1].departure_day > bay[i].departure_day:
                inversions += 1
    return inversions


def _adjacent_hazmat_violation(yard: YardState, bay_idx: int) -> bool:
    def bay_has_hazmat(b_idx: int) -> bool:
        if 0 <= b_idx < NUM_BAYS:
            return any(c.is_hazmat for c in yard.bays[b_idx])
        return False

    return bay_has_hazmat(bay_idx - 1) or bay_has_hazmat(bay_idx + 1)


# ─────────────────────────────────────────────
# MAIN ENVIRONMENT
# ─────────────────────────────────────────────


class PortOpsEnv:
    def __init__(self):
        self._yard: YardState = YardState()
        self._inbound_queue: List[Container] = []
        self._outbound_requests: List[str] = []
        self._step_count: int = 0
        self._task_id: int = 1
        self._seed: int = 42
        self._done: bool = False
        self._fatal_error: bool = False
        self._last_error: Optional[str] = None
        self._target_container_id: Optional[str] = None
        self._opt_moves: int = 0
        self._task3_inbound_placed: int = 0
        self._task3_required_placed: int = 2

    def reset(self, task_id: int = 1, seed: int = 42) -> ObservationSpace:
        self._task_id = task_id
        self._seed = seed
        self._step_count = 0
        self._done = False
        self._fatal_error = False
        self._last_error = None
        self._task3_inbound_placed = 0
        rng = random.Random(seed)

        if task_id == 1:
            self._setup_task1(rng)
        elif task_id == 2:
            self._setup_task2(rng)
        elif task_id == 3:
            self._setup_task3(rng)

        return self._build_observation()

    def _setup_task1(self, rng: random.Random):
        containers = [Container(id=f"C{i:02d}") for i in range(1, 6)]
        bays = [[] for _ in range(NUM_BAYS)]
        bays[0] = [containers[0], containers[1], containers[2]]
        bays[1] = [containers[3]]
        bays[2] = [containers[4]]
        self._yard = YardState(bays=bays)
        self._target_container_id = "C01"
        self._outbound_requests = ["C01"]
        self._opt_moves = _compute_min_moves_extraction(self._yard, "C01")

    def _setup_task2(self, rng: random.Random):
        self._yard = YardState()
        self._inbound_queue = [
            Container(id=f"C{i:02d}", departure_day=rng.randint(1, 5))
            for i in range(1, 9)
        ]
        self._outbound_requests = []

    def _setup_task3(self, rng: random.Random):
        def make_c(cid, w, hz, dep):
            return Container(id=cid, weight=w, is_hazmat=hz, departure_day=dep)

        bays = [[] for _ in range(NUM_BAYS)]
        bays[0] = [
            make_c("C01", "Heavy", False, 2),
            make_c("C02", "Light", False, 3),
            make_c("C03", "Light", False, 5),
        ]
        bays[1] = [
            make_c("C04", "Light", True, 1),
            make_c("C05", "Heavy", False, 4),
            make_c("C06", "Light", False, 2),
        ]
        bays[2] = [
            make_c("C07", "Light", False, 1),
            make_c("C08", "Heavy", False, 3),
        ]
        bays[3] = [
            make_c("C09", "Light", False, 2),
            make_c("C10", "Heavy", False, 5),
        ]
        bays[4] = [make_c("C11", "Light", True, 3), make_c("C12", "Heavy", False, 4)]
        self._yard = YardState(bays=bays)
        self._inbound_queue = [
            make_c("C13", "Heavy", False, 3),
            make_c("C14", "Light", True, 2),
        ]
        self._target_container_id = "C07"
        self._outbound_requests = ["C07"]

    def state(self) -> Dict[str, Any]:
        yard_dict = {}
        for i, bay in enumerate(self._yard.bays):
            yard_dict[f"bay_{i + 1}"] = [c.model_dump() for c in bay]

        return {
            "yard": yard_dict,
            "inbound_queue": [c.model_dump() for c in self._inbound_queue],
            "outbound_requests": list(self._outbound_requests),
            "step_count": self._step_count,
            "task_id": self._task_id,
            "done": self._done,
            "fatal_error": self._fatal_error,
            "last_error": self._last_error,
        }

    def step(
        self, action: ActionSpace
    ) -> Tuple[ObservationSpace, float, bool, Dict[str, Any]]:
        if self._done:
            score = self._compute_final_score()
            return (
                self._build_observation(),
                score,
                True,
                {"reason": "Done", "score": score},
            )

        try:
            action_type, args = action.parse()
            if action_type == "move":
                self._last_error = self._execute_move(args[0], int(args[1]) - 1)
            else:
                self._last_error = self._execute_retrieve(args[0])
        except ValueError as e:
            self._last_error = str(e)

        self._step_count += 1

        if self._fatal_error:
            self._done = True
            score = self._compute_final_score()
            return (
                self._build_observation(),
                score,
                True,
                {"reason": "Fatal error", "score": score},
            )

        if self._is_task_complete() or self._step_count >= MAX_STEPS:
            self._done = True
            score = self._compute_final_score()
            return self._build_observation(), score, True, {"score": score}

        return self._build_observation(), 0.0, False, {"step": self._step_count}

    def _execute_move(self, cid: str, target_idx: int) -> Optional[str]:
        if not (0 <= target_idx < NUM_BAYS):
            return "Invalid bay"

        container = None
        from_inbound = False
        source_idx: Optional[int] = None

        if self._inbound_queue and self._inbound_queue[0].id == cid:
            container = self._inbound_queue[0]
            from_inbound = True
        else:
            loc = self._yard.find_container(cid)
            if not loc:
                return f"{cid} not found"
            if not self._yard.is_accessible(cid):
                return f"{cid} blocked"
            source_idx = loc[0]
            container = self._yard.bays[loc[0]][-1]

        if len(self._yard.bays[target_idx]) >= MAX_TIERS:
            return "Bay full"

        # Task 3 Constraints
        if self._task_id == 3:
            top = self._yard.top_container(target_idx)
            if top and container.weight == "Heavy" and top.weight == "Light":
                self._fatal_error = True
                return "FATAL: Heavy on Light"
            if container.is_hazmat and _adjacent_hazmat_violation(
                self._yard, target_idx
            ):
                self._fatal_error = True
                return "FATAL: Hazmat adjacency"

        if from_inbound:
            self._inbound_queue.pop(0)
            if self._task_id == 3:
                self._task3_inbound_placed += 1
        else:
            assert source_idx is not None
            self._yard.bays[source_idx].pop()

        self._yard.bays[target_idx].append(container)
        return None

    def _execute_retrieve(self, cid: str) -> Optional[str]:
        if cid not in self._outbound_requests:
            return "Not requested"
        loc = self._yard.find_container(cid)
        if not loc:
            return "Not in yard"
        if not self._yard.is_accessible(cid):
            return "Blocked"

        self._yard.bays[loc[0]].pop()
        self._outbound_requests.remove(cid)
        return None

    def _is_task_complete(self) -> bool:
        if self._task_id == 1:
            return not self._outbound_requests
        if self._task_id == 2:
            return not self._inbound_queue
        if self._task_id == 3:
            return not self._outbound_requests and self._task3_inbound_placed >= 2
        return False

    def _compute_final_score(self) -> float:
        score = MIN_SCORE
        if not self._fatal_error:
            if self._task_id == 1:
                score = self._grade_task1()
            elif self._task_id == 2:
                score = self._grade_task2()
            elif self._task_id == 3:
                score = self._grade_task3()
        return max(MIN_SCORE, min(MAX_SCORE, score))

    def _grade_task1(self) -> float:
        if self._outbound_requests:
            return MIN_SCORE
        return max(
            MIN_SCORE,
            min(MAX_SCORE, 1.0 - 0.2 * (self._step_count - self._opt_moves)),
        )

    def _grade_task2(self) -> float:
        inv = _count_temporal_inversions(self._yard)
        return max(MIN_SCORE, min(MAX_SCORE, 1.0 - 0.15 * inv))

    def _grade_task3(self) -> float:
        if not self._is_task_complete():
            return MIN_SCORE
        return max(MIN_SCORE, min(MAX_SCORE, 1.0 - 0.1 * (self._step_count - 4)))

    def _compute_safe_moves(self) -> List[str]:
        safe = []
        for cid in self._outbound_requests:
            if self._yard.is_accessible(cid):
                safe.append(f"retrieve({cid})")

        movable = []
        if self._inbound_queue:
            movable.append(self._inbound_queue[0])
        for bay in self._yard.bays:
            if bay:
                movable.append(bay[-1])

        for container in movable:
            for target_idx in range(NUM_BAYS):
                loc = self._yard.find_container(container.id)
                if loc and loc[0] == target_idx:
                    continue
                if len(self._yard.bays[target_idx]) >= MAX_TIERS:
                    continue

                if self._task_id == 3:
                    top = self._yard.top_container(target_idx)
                    if top and container.weight == "Heavy" and top.weight == "Light":
                        continue
                    if container.is_hazmat and _adjacent_hazmat_violation(
                        self._yard, target_idx
                    ):
                        continue
                safe.append(f"move({container.id},{target_idx + 1})")
        return safe

    def _build_observation(self) -> ObservationSpace:
        yard_text = self._yard.bay_summary()
        if self._task_id == 3:
            safe_moves = self._compute_safe_moves()
            yard_text += f"\n\n✅ SAFE MOVES: {', '.join(safe_moves)}"

        return ObservationSpace(
            yard_text=yard_text,
            inbound_queue=[c.id for c in self._inbound_queue],
            outbound_requests=list(self._outbound_requests),
            last_action_error=self._last_error,
            step_count=self._step_count,
            steps_remaining=max(0, MAX_STEPS - self._step_count),
        )


# Singleton
_env_instance = PortOpsEnv()


def get_env() -> PortOpsEnv:
    return _env_instance
