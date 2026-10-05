"""Evaluate a fixed database policy with synchronous Bellman expectation updates."""

import json
from math import isclose, isfinite
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import open_engine, reflected_models


def evaluate_policy(path: Path, policy_id: int = 1,
                    tolerance: float = 1e-10, max_iterations: int = 10000, *,
                    policy_probabilities: dict[tuple[int, int], float] | None = None) -> dict:
    """Return values and iteration history without modifying the database.

    Each sweep uses only values from the preceding sweep. The terminal value
    stays zero; the reward on a transition into it is still included.
    tolerance is the maximum change between successive sweeps, not a promise
    of that same absolute error in the returned values.
    An optional in-memory policy replaces the stored action probabilities for
    this evaluation only. Its result has no database policy ID.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    engine = open_engine(path)
    try:
        base = reflected_models(engine)
        with Session(engine) as session:
            Environment = base.classes.environment
            State = base.classes.state
            Policy = base.classes.policy
            Transition = base.classes.transition
            PolicyProbability = base.classes.policy_probability
            environment = session.get(Environment, 1)
            policy = session.get(Policy, policy_id)
            if environment is None or policy is None:
                raise ValueError("The environment or requested policy is missing.")
            discount = json.loads(environment.specification_json)["discount"]
            initial_state_id = environment.initial_state_id
            policy_name = policy.name
            states = session.execute(select(
                State.id, State.gap_m, State.speed_mps, State.is_terminal,
            ).order_by(State.id)).all()
            transitions = session.execute(select(
                Transition.state_id, Transition.action_id, Transition.next_state_id,
                Transition.transition_probability, Transition.reward,
            ).order_by(Transition.state_id, Transition.action_id)).all()
            policy_probability_rows = session.execute(select(
                PolicyProbability.state_id, PolicyProbability.action_id,
                PolicyProbability.policy_probability,
            ).where(PolicyProbability.policy_id == policy_id)).all()
    finally:
        engine.dispose()

    if not isfinite(discount) or not 0 <= discount < 1:
        raise ValueError("Discount must be finite and satisfy 0 <= discount < 1.")
    values = {state.id: 0.0 for state in states}
    normal_ids = {state.id for state in states if not state.is_terminal}
    terminal_ids = {state.id for state in states if state.is_terminal}
    if terminal_ids != {0} or initial_state_id not in normal_ids:
        raise ValueError("Expected terminal state 0 and a nonterminal initial state.")
    if policy_probabilities is None:
        policy_probabilities = {
            (current_state_id, action_id): policy_probability
            for current_state_id, action_id, policy_probability in policy_probability_rows
        }
    else:
        policy_probabilities = dict(policy_probabilities)
        policy_id = None
        policy_name = "in_memory_policy"
    if set(policy_probabilities) != {(t.state_id, t.action_id) for t in transitions}:
        raise ValueError("Policy entries must cover every state-action transition.")
    policy_probability_sums = dict.fromkeys(normal_ids, 0.0)
    evaluation_transitions = []
    for current_state_id, action_id, next_state_id, transition_probability, reward in transitions:
        policy_probability = policy_probabilities[current_state_id, action_id]
        if current_state_id not in normal_ids or next_state_id not in values:
            raise ValueError("Transition references an invalid state.")
        if transition_probability != 1.0 or not isfinite(reward):
            raise ValueError("Expected finite rewards and deterministic transitions.")
        if not isfinite(policy_probability) or not 0 <= policy_probability <= 1:
            raise ValueError("Invalid action probability.")
        policy_probability_sums[current_state_id] += policy_probability
        evaluation_transitions.append(
            (current_state_id, next_state_id, policy_probability, transition_probability, reward)
        )
    if any(not isclose(total, 1.0, rel_tol=0, abs_tol=1e-12)
           for total in policy_probability_sums.values()):
        raise ValueError("Action probabilities must sum to 1 at every nonterminal state.")

    def bellman_update(old_values):
        new_values = dict.fromkeys(old_values, 0.0)
        for current_state_id, next_state_id, policy_probability, transition_probability, reward in evaluation_transitions:
            new_values[current_state_id] += (
                policy_probability * transition_probability
                * (reward + discount * old_values[next_state_id])
            )
        return new_values

    history = []
    for iteration in range(1, max_iterations + 1):
        new_values = bellman_update(values)
        max_change = max(
            abs(new_values[state_id] - values[state_id]) for state_id in normal_ids
        )
        values = new_values
        history.append({"iteration": iteration, "max_change": max_change,
                        "initial_state_value": values[initial_state_id]})
        if max_change <= tolerance:
            break
    else:
        raise RuntimeError(f"Policy evaluation did not converge in {max_iterations} iterations.")

    bellman_values = bellman_update(values)
    bellman_residual = max(
        abs(bellman_values[state_id] - values[state_id]) for state_id in normal_ids
    )
    return {
        "summary": {
            "policy_id": policy_id, "policy_name": policy_name, "discount": discount,
            "iterations": iteration, "tolerance": tolerance, "max_change": max_change,
            "bellman_residual": bellman_residual,
            "value_error_bound": bellman_residual / (1 - discount),
            "initial_state_id": initial_state_id, "initial_state_value": values[initial_state_id],
            "terminal_value": values[0], "converged": True,
        },
        "values": [{"state_id": s.id, "gap_m": s.gap_m, "speed_mps": s.speed_mps,
                    "is_terminal": s.is_terminal, "value": values[s.id]} for s in states],
        "history": history,
    }
