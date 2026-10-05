"""Exact straight-line rules shared by SQLite generation and the HighwayEnv adapter."""

from dataclasses import asdict, dataclass
from math import isclose, sqrt


@dataclass(frozen=True)
class Specification:
    lead_speed_mps: int = 20
    decision_seconds: int = 1
    acceleration_mps2: int = 6
    max_speed_mps: int = 40
    max_gap_m: int = 200
    desired_gap_min_m: int = 40
    desired_gap_max_m: int = 80
    initial_gap_m: int = 50
    initial_speed_mps: int = 20
    discount: float = 0.9
    demonstration_steps: int = 20
    collision_reward: int = -100
    lost_reward: int = -50
    stopped_reward: int = -20
    overspeed_reward: int = -10
    close_reward: int = -1
    desired_reward: int = 1
    distant_reward: int = 0

    def as_dict(self):
        return asdict(self)


SPEC = Specification()
TERMINAL_ID = 0
# Action IDs are shared by the database and the simulation adapter.
ACTIONS = ((0, "accelerate", 6), (1, "maintain", 0), (2, "decelerate", -6))
TOLERANCE = 1e-10


def state_id(gap_m: int, speed_mps: int) -> int:
    if not (isinstance(gap_m, int) and isinstance(speed_mps, int)):
        raise ValueError("Nonterminal state coordinates must be integers.")
    if not (1 <= gap_m <= SPEC.max_gap_m and 1 <= speed_mps <= SPEC.max_speed_mps):
        raise ValueError("Coordinates are outside the nonterminal state space.")
    return (gap_m - 1) * SPEC.max_speed_mps + speed_mps


@dataclass(frozen=True)
class Outcome:
    next_state_id: int
    reward: int
    reason: str | None
    elapsed_seconds: float
    gap_after_m: float
    speed_after_mps: float

    @property
    def terminated(self) -> bool:
        return self.next_state_id == TERMINAL_ID


def _roots(quadratic: float, linear: float, constant: float) -> list[float]:
    if quadratic == 0:
        return [] if linear == 0 else [-constant / linear]
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < -TOLERANCE:
        return []
    radical = sqrt(max(0.0, discriminant))
    return [(-linear - radical) / (2 * quadratic), (-linear + radical) / (2 * quadratic)]


def transition(gap_m: int, speed_mps: int, action_id: int) -> Outcome:
    """Advance up to one second, stopping at the first terminal boundary.

    A collision wins a simultaneous tie. The strict upper limits (>200 m,
    >40 m/s) terminate only when motion would pass the boundary before the
    decision interval ends. Their reported time/coordinates are the limiting
    crossing values; merely ending at exactly 200 m or 40 m/s is allowed.
    """
    state_id(gap_m, speed_mps)
    if action_id not in (0, 1, 2):
        raise ValueError("Action ID must be 0, 1, or 2.")
    acceleration = ACTIONS[action_id][2]
    duration = SPEC.decision_seconds
    relative_speed = SPEC.lead_speed_mps - speed_mps
    quadratic = -0.5 * acceleration
    events = []

    for time in _roots(quadratic, relative_speed, gap_m):
        if -TOLERANCE <= time <= duration + TOLERANCE:
            events.append((max(0.0, min(duration, time)), 0, "collision"))

    for time in _roots(quadratic, relative_speed, gap_m - SPEC.max_gap_m):
        slope = relative_speed + 2 * quadratic * time
        crosses_outward = slope > TOLERANCE or (
            abs(time) <= TOLERANCE and abs(slope) <= TOLERANCE and quadratic > 0
        )
        if crosses_outward and -TOLERANCE <= time < duration - TOLERANCE:
            events.append((max(0.0, time), 1, "lost"))

    if acceleration < 0:
        stop_time = -speed_mps / acceleration
        if stop_time <= duration + TOLERANCE:
            events.append((min(duration, stop_time), 2, "stopped"))
    if acceleration > 0:
        limit_time = (SPEC.max_speed_mps - speed_mps) / acceleration
        if limit_time < duration - TOLERANCE:
            events.append((max(0.0, limit_time), 3, "overspeed"))

    reason = None
    elapsed = float(duration)
    if events:
        first_time = min(item[0] for item in events)
        tied = [item for item in events if abs(item[0] - first_time) <= TOLERANCE]
        elapsed, _, reason = min(tied, key=lambda item: item[1])

    new_speed = speed_mps + acceleration * elapsed
    new_gap = gap_m + relative_speed * elapsed + quadratic * elapsed * elapsed
    if reason is not None:
        return Outcome(
            TERMINAL_ID, getattr(SPEC, f"{reason}_reward"), reason,
            elapsed, new_gap, new_speed,
        )

    # Full-second transitions are exactly integer-valued for this specification.
    gap_integer, speed_integer = round(new_gap), round(new_speed)
    if not (isclose(new_gap, gap_integer, abs_tol=TOLERANCE)
            and isclose(new_speed, speed_integer, abs_tol=TOLERANCE)):
        raise ArithmeticError("Motion left the exact integer grid; do not silently quantize.")
    next_id = state_id(gap_integer, speed_integer)
    if new_gap < SPEC.desired_gap_min_m:
        reward = SPEC.close_reward
    elif new_gap <= SPEC.desired_gap_max_m:
        reward = SPEC.desired_reward
    else:
        reward = SPEC.distant_reward
    return Outcome(next_id, reward, None, elapsed, new_gap, new_speed)
