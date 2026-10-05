"""Run the existing MDP and policies with HighwayEnv roads, vehicles and rendering."""

from contextlib import closing
import json
from math import isclose
from pathlib import Path
import sqlite3

import gymnasium as gym
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from .improvement import iterate_policy
from .mdp import ACTIONS, SPEC, TOLERANCE, state_id, transition


class ExactVehicle(Vehicle):
    """Keep HighwayEnv's vehicle representation with exact straight-line motion."""

    def step(self, dt: float) -> None:
        acceleration = self.action["acceleration"]
        self.position[0] += self.speed * dt + 0.5 * acceleration * dt * dt
        self.speed += acceleration * dt
        self.on_state_update()


class CarFollowingEnv(AbstractEnv):
    """Two vehicles on one lane, using the agreed MDP actions and failure rules.

    Observations are [bumper gap in metres, following speed in metres/second].
    The analytic MDP finds the first failure time and reward. HighwayEnv's Road
    advances the actual vehicles up to that time; their measured coordinates
    must match the MDP. No stock speed controller or collision response is used.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "simulation_frequency": 5, "policy_frequency": 1,
            "screen_width": 900, "screen_height": 180,
            "scaling": 3.5, "centering_position": [0.12, 0.5],
            "offscreen_rendering": True,
        })
        return config

    def define_spaces(self) -> None:
        # Action IDs match SQLite: accelerate, maintain, decelerate.
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        gap = options.get("gap_m", SPEC.initial_gap_m)
        speed = options.get("speed_mps", SPEC.initial_speed_mps)
        self.current_state_id = state_id(gap, speed)
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=1, start=-100, length=10000, speed_limit=SPEC.max_speed_mps,
            ),
            np_random=self.np_random,
        )
        follower = ExactVehicle(self.road, [0, 0], speed=speed)
        # HighwayEnv positions are vehicle centres; the MDP gap is bumper-to-bumper.
        self.lead_vehicle = ExactVehicle(
            self.road, [gap + follower.LENGTH, 0], speed=SPEC.lead_speed_mps,
        )
        follower.color = (60, 130, 245)
        self.lead_vehicle.color = (60, 190, 100)
        self.controlled_vehicles = [follower]
        self.road.vehicles = [follower, self.lead_vehicle]
        for vehicle in self.road.vehicles:
            # The MDP determines exact contact; disable predictive stock impacts.
            vehicle.check_collisions = False
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.last_outcome = None
        self.frames = []
        self.frame_times = []
        return self._observation(), {"state_id": self.current_state_id, "time_seconds": self.time}

    def _observation(self) -> np.ndarray:
        gap = (self.lead_vehicle.position[0] - self.vehicle.position[0]
               - (self.lead_vehicle.LENGTH + self.vehicle.LENGTH) / 2)
        return np.array([gap, self.vehicle.speed], dtype=np.float64)

    def render(self):
        # Keep both vehicles visible, with a closer view for ordinary following gaps.
        gap = max(0.0, self._observation()[0])
        scaling = min(8.0, self.config["screen_width"] * 0.8 / (gap + 2 * self.vehicle.LENGTH))
        self.config["scaling"] = scaling
        if self.viewer is not None:
            self.viewer.sim_surface.scaling = scaling
        return super().render()

    def step(self, action):
        if self.done:
            raise RuntimeError("The episode has ended; call reset before another step.")
        if not self.action_space.contains(action):
            raise ValueError("Action ID must be 0, 1, or 2.")
        gap, speed = self._observation()
        gap_integer, speed_integer = round(gap), round(speed)
        if not (isclose(gap, gap_integer, rel_tol=0, abs_tol=TOLERANCE)
                and isclose(speed, speed_integer, rel_tol=0, abs_tol=TOLERANCE)):
            raise ArithmeticError("The simulated decision state left the integer grid.")
        outcome = transition(gap_integer, speed_integer, int(action))
        self.vehicle.act({"steering": 0, "acceleration": ACTIONS[int(action)][2]})
        self.lead_vehicle.act({"steering": 0, "acceleration": 0})
        self.frames, self.frame_times = [], []
        start_time = self.time
        elapsed = 0.0
        while elapsed < outcome.elapsed_seconds:
            dt = min(1 / self.config["simulation_frequency"], outcome.elapsed_seconds - elapsed)
            self.road.step(dt)
            elapsed += dt
            self.time = start_time + elapsed
            if self.render_mode == "rgb_array":
                self.frames.append(self.render())
                self.frame_times.append(self.time)
        observation = self._observation()
        if not np.allclose(observation, [outcome.gap_after_m, outcome.speed_after_mps],
                           rtol=0, atol=TOLERANCE):
            raise ArithmeticError("HighwayEnv vehicle coordinates disagree with the MDP.")
        self.last_outcome = outcome
        self.steps += 1
        terminated = outcome.terminated
        # The 20-decision cap ends the demonstration, not the underlying MDP.
        truncated = self.steps >= SPEC.demonstration_steps and not terminated
        self.done = terminated or truncated
        self.current_state_id = (
            0 if terminated else state_id(round(observation[0]), round(observation[1]))
        )
        if outcome.reason == "collision":
            self.vehicle.crashed = self.lead_vehicle.crashed = True
            if self.frames:
                self.frames[-1] = self.render()
        info = {
            "state_id": self.current_state_id, "terminal_reason": outcome.reason,
            "elapsed_seconds": outcome.elapsed_seconds, "time_seconds": self.time,
        }
        return observation, float(outcome.reward), terminated, truncated, info


def run_simulation(path: Path, *, policy: str = "final", seed: int = 0,
                   capture_frames: bool = False, iteration_result: dict | None = None) -> dict:
    """Sample initial/final policy actions for at most 20 decisions; never write SQLite.

    Supplying the existing Notebook's iteration_result avoids repeating policy
    iteration. Every observed transition is also checked against its SQL record.
    """
    if policy not in ("initial", "final"):
        raise ValueError("policy must be 'initial' or 'final'.")
    path = Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        specification, initial_id = connection.execute(
            "SELECT specification_json, initial_state_id FROM environment WHERE id = 1"
        ).fetchone()
        if (json.loads(specification) != SPEC.as_dict()
                or initial_id != state_id(SPEC.initial_gap_m, SPEC.initial_speed_mps)
                or connection.execute("SELECT * FROM action ORDER BY id").fetchall() != list(ACTIONS)):
            raise ValueError("Database settings do not match the simulation specification.")
        connection.row_factory = sqlite3.Row
        transitions = {
            (row["state_id"], row["action_id"]): dict(row)
            for row in connection.execute("SELECT * FROM transition")
        }
        policy_probabilities = {
            (row["state_id"], row["action_id"]): row["policy_probability"]
            for row in connection.execute("SELECT * FROM policy_probability WHERE policy_id = 1")
        }
    if policy == "final":
        iteration_result = iteration_result or iterate_policy(path)
        policy_probabilities = {
            (row["state_id"], row["action_id"]): row["final_policy_probability"]
            for row in iteration_result["action_values"]
        }
    if set(policy_probabilities) != set(transitions):
        raise ValueError("The policy must cover every stored state-action pair.")
    for current_state in {key[0] for key in transitions}:
        probabilities = [policy_probabilities[current_state, action] for action, _, _ in ACTIONS]
        if (not all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                or not isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Invalid policy probabilities.")

    env = CarFollowingEnv(render_mode="rgb_array" if capture_frames else None)
    rng = np.random.default_rng(seed)
    trajectory, frames, frame_times = [], [], []
    discounted_return = 0.0
    try:
        observation, info = env.reset(seed=seed)
        if capture_frames:
            frames.append(env.render())
            frame_times.append(0.0)
        for decision in range(1, SPEC.demonstration_steps + 1):
            current_state = info["state_id"]
            before = observation.copy()
            probabilities = [policy_probabilities[current_state, action] for action, _, _ in ACTIONS]
            action = int(rng.choice(len(ACTIONS), p=probabilities))
            observation, reward, terminated, truncated, info = env.step(action)
            expected = transitions[current_state, action]
            if (expected["next_state_id"] != info["state_id"]
                    or expected["terminal_reason"] != info["terminal_reason"]
                    or expected["reward"] != reward or expected["transition_probability"] != 1
                    or not np.allclose(
                        [*observation, info["elapsed_seconds"]],
                        [expected["gap_after_m"], expected["speed_after_mps"], expected["elapsed_seconds"]],
                        rtol=0, atol=TOLERANCE)):
                raise ArithmeticError("The simulated transition disagrees with SQLite.")
            discounted_return += SPEC.discount ** (decision - 1) * reward
            trajectory.append({
                "decision": decision, "state_id": current_state,
                "gap_before_m": float(before[0]), "speed_before_mps": float(before[1]),
                "action": ACTIONS[action][1], "policy_probability": probabilities[action],
                "reward": reward, "next_state_id": info["state_id"],
                "gap_after_m": float(observation[0]), "speed_after_mps": float(observation[1]),
                "elapsed_seconds": info["elapsed_seconds"], "time_seconds": info["time_seconds"],
                "terminated": terminated, "truncated": truncated,
                "terminal_reason": info["terminal_reason"],
            })
            frames.extend(env.frames)
            frame_times.extend(env.frame_times)
            if terminated or truncated:
                break
        return {
            "summary": {
                "policy": policy, "seed": seed, "decisions": len(trajectory),
                "total_reward": sum(row["reward"] for row in trajectory),
                "sample_discounted_return": discounted_return,
                "terminated": terminated, "truncated": truncated,
                "terminal_reason": info["terminal_reason"],
                "time_seconds": info["time_seconds"], "sql_transitions_match": True,
            },
            "trajectory": trajectory, "frames": frames, "frame_times": frame_times,
        }
    finally:
        env.close()
