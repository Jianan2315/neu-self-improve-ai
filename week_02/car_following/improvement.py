"""One greedy policy improvement, or repeated evaluation and improvement."""

from math import isclose, isfinite
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import open_engine, reflected_models
from .evaluation import evaluate_policy


def greedy_policy_probabilities(action_values: list[dict],
                                action_value_tie_tolerance: float = 1e-10) -> dict:
    """Split probability equally among actions within an absolute tie tolerance."""
    if not isfinite(action_value_tie_tolerance) or action_value_tie_tolerance < 0:
        raise ValueError("action_value_tie_tolerance must be nonnegative and finite.")
    actions_by_state = {}
    for row in action_values:
        if not isfinite(row["action_value"]):
            raise ValueError("Action values must be finite.")
        actions_by_state.setdefault(row["state_id"], []).append(row)

    policy_probabilities = {}
    for current_state_id, actions in actions_by_state.items():
        best_action_value = max(row["action_value"] for row in actions)
        best_action_ids = {
            row["action_id"] for row in actions
            if isclose(row["action_value"], best_action_value,
                       rel_tol=0, abs_tol=action_value_tie_tolerance)
        }
        for row in actions:
            policy_probabilities[current_state_id, row["action_id"]] = (
                1.0 / len(best_action_ids) if row["action_id"] in best_action_ids else 0.0
            )
    return policy_probabilities


def _read_policy_transitions(path: Path) -> list:
    """Read transitions and policy 1 probabilities for both experiments."""
    engine = open_engine(Path(path))
    try:
        base = reflected_models(engine)
        with Session(engine) as session:
            Transition = base.classes.transition
            Action = base.classes.action
            PolicyProbability = base.classes.policy_probability
            transitions = session.execute(select(
                Transition.state_id, Transition.action_id, Action.name,
                Transition.next_state_id, Transition.transition_probability,
                Transition.reward, PolicyProbability.policy_probability,
            ).join(Action, Action.id == Transition.action_id).join(
                PolicyProbability,
                (PolicyProbability.state_id == Transition.state_id)
                & (PolicyProbability.action_id == Transition.action_id),
            ).where(PolicyProbability.policy_id == 1)
                .order_by(Transition.state_id, Transition.action_id)).all()
    finally:
        engine.dispose()
    return transitions


def _greedy_improvement_step(transitions: list, values: dict,
                             current_policy_probabilities: dict,
                             discount: float, action_value_tie_tolerance: float) -> dict:
    """Improve an already evaluated policy; shared by both public experiments."""
    action_values = []
    for row in transitions:
        action_values.append({
            "state_id": row.state_id, "action_id": row.action_id, "action": row.name,
            "action_value": row.transition_probability * (
                row.reward + discount * values[row.next_state_id]
            ),
        })
    next_policy_probabilities = greedy_policy_probabilities(action_values, action_value_tie_tolerance)
    changed_state_ids = {
        state_id for state_id, action_id in current_policy_probabilities
        if not isclose(current_policy_probabilities[state_id, action_id],
                       next_policy_probabilities[state_id, action_id],
                       rel_tol=0, abs_tol=1e-12)
    }
    return {
        "action_values": action_values,
        "policy_probabilities": next_policy_probabilities,
        "changed_state_ids": changed_state_ids,
    }


def improve_policy(path: Path, tolerance: float = 1e-10,
                   max_iterations: int = 10000) -> dict:
    """Evaluate policy 1, improve it once, and evaluate the new in-memory policy.

    Action values use the INITIAL policy's state values. The improved policy is
    held fixed during its evaluation. No policy or value is written to SQLite.
    One tolerance controls both evaluation convergence and action-value ties.
    """
    state_value_convergence_tolerance = tolerance
    action_value_tie_tolerance = tolerance
    initial_evaluation = evaluate_policy(
        path, tolerance=state_value_convergence_tolerance, max_iterations=max_iterations,
    )
    initial_values = {row["state_id"]: row["value"] for row in initial_evaluation["values"]}
    discount = initial_evaluation["summary"]["discount"]
    transitions = _read_policy_transitions(path)

    initial_policy_probabilities = {
        (row.state_id, row.action_id): row.policy_probability for row in transitions
    }
    improvement = _greedy_improvement_step(
        transitions, initial_values, initial_policy_probabilities, discount, action_value_tie_tolerance,
    )
    action_values = improvement["action_values"]
    improved_policy_probabilities = improvement["policy_probabilities"]
    changed_state_ids = improvement["changed_state_ids"]
    for row, transition in zip(action_values, transitions):
        row.update({
            "next_state_id": transition.next_state_id,
            "transition_probability": transition.transition_probability,
            "reward": transition.reward,
            "initial_next_state_value": initial_values[transition.next_state_id],
            "initial_policy_probability": transition.policy_probability,
            "improved_policy_probability": improved_policy_probabilities[
                transition.state_id, transition.action_id
            ],
        })

    improved_evaluation = evaluate_policy(
        path, tolerance=state_value_convergence_tolerance, max_iterations=max_iterations,
        policy_probabilities=improved_policy_probabilities,
    )
    improved_evaluation["summary"]["policy_name"] = "greedy_once_from_initial"
    value_comparison = []
    for row in improved_evaluation["values"]:
        initial_value = initial_values[row["state_id"]]
        value_comparison.append({
            "state_id": row["state_id"], "gap_m": row["gap_m"],
            "speed_mps": row["speed_mps"], "is_terminal": row["is_terminal"],
            "initial_value": initial_value, "improved_value": row["value"],
            "value_gain": row["value"] - initial_value,
        })

    initial_state_id = initial_evaluation["summary"]["initial_state_id"]
    initial_value = initial_values[initial_state_id]
    improved_value = improved_evaluation["summary"]["initial_state_value"]
    # Account for both approximate evaluations and the numerical tie tolerance.
    comparison_tolerance = (
        initial_evaluation["summary"]["value_error_bound"]
        + improved_evaluation["summary"]["value_error_bound"]
        + action_value_tie_tolerance / (1 - discount)
    )
    return {
        "summary": {
            "improvement_rounds": 1, "initial_state_id": initial_state_id,
            "discount": discount, "action_value_tie_tolerance": action_value_tie_tolerance,
            "state_value_convergence_tolerance": state_value_convergence_tolerance,
            "changed_states": len(changed_state_ids),
            "initial_state_value_before": initial_value,
            "initial_state_value_after": improved_value,
            "initial_state_value_gain": improved_value - initial_value,
            "minimum_value_gain": min(row["value_gain"] for row in value_comparison
                                      if not row["is_terminal"]),
            "comparison_tolerance": comparison_tolerance,
            "states_with_lower_value": sum(
                row["value_gain"] < -comparison_tolerance for row in value_comparison
                if not row["is_terminal"]
            ),
        },
        "initial_evaluation": initial_evaluation,
        "improved_evaluation": improved_evaluation,
        "action_values": action_values,
        "value_comparison": value_comparison,
    }


def iterate_policy(path: Path, tolerance: float = 1e-10,
                   max_iterations: int = 10000, max_improvements: int = 100) -> dict:
    """Repeat fixed-policy evaluation and greedy improvement, without DB writes.

    History round 0 evaluates the stored initial policy. Each row reports the
    changes proposed AFTER evaluating that round's policy. Stop only when the
    probabilities are stable and the Bellman optimality residual is small.
    max_iterations limits evaluation sweeps; max_improvements limits policy changes.
    One tolerance controls both evaluation convergence and action-value ties.
    """
    state_value_convergence_tolerance = tolerance
    action_value_tie_tolerance = tolerance
    if not isinstance(max_improvements, int) or max_improvements < 1:
        raise ValueError("max_improvements must be a positive integer.")

    initial_evaluation = evaluate_policy(
        path, tolerance=state_value_convergence_tolerance, max_iterations=max_iterations,
    )
    transitions = _read_policy_transitions(path)
    initial_policy_probabilities = {
        (row.state_id, row.action_id): row.policy_probability for row in transitions
    }
    current_policy_probabilities = dict(initial_policy_probabilities)
    current_evaluation = initial_evaluation
    discount = initial_evaluation["summary"]["discount"]
    # At stability: the best-action gap is bounded by tie tolerance plus
    # the fixed-policy evaluation residual, bounded by state_value_convergence_tolerance.
    optimality_tolerance = action_value_tie_tolerance + state_value_convergence_tolerance
    history = []

    for improvement_round in range(max_improvements + 1):
        values = {row["state_id"]: row["value"] for row in current_evaluation["values"]}
        improvement = _greedy_improvement_step(
            transitions, values, current_policy_probabilities, discount, action_value_tie_tolerance,
        )
        action_values = improvement["action_values"]
        next_policy_probabilities = improvement["policy_probabilities"]
        changed_state_ids = improvement["changed_state_ids"]
        best_action_values = {}
        for row in action_values:
            best_action_values[row["state_id"]] = max(
                best_action_values.get(row["state_id"], float("-inf")), row["action_value"],
            )
        optimality_residual = max(
            abs(best_action_values[state_id] - values[state_id])
            for state_id in best_action_values
        )
        history.append({
            "improvement_round": improvement_round,
            "evaluation_sweeps": current_evaluation["summary"]["iterations"],
            "initial_state_value": current_evaluation["summary"]["initial_state_value"],
            "changed_states": len(changed_state_ids),
            "policy_residual": current_evaluation["summary"]["bellman_residual"],
            "optimality_residual": optimality_residual,
        })
        if not changed_state_ids and optimality_residual <= optimality_tolerance:
            break
        if improvement_round == max_improvements:
            raise RuntimeError("Policy iteration did not converge within max_improvements.")
        current_policy_probabilities = next_policy_probabilities
        current_evaluation = evaluate_policy(
            path, tolerance=state_value_convergence_tolerance, max_iterations=max_iterations,
            policy_probabilities=current_policy_probabilities,
        )
    else:
        raise RuntimeError("Policy iteration did not converge.")

    # Copy the summary so an initially stable policy cannot rename the initial result.
    final_evaluation = {**current_evaluation, "summary": {
        **current_evaluation["summary"], "policy_name": "policy_iteration_final",
    }}
    initial_values = {row["state_id"]: row["value"] for row in initial_evaluation["values"]}
    for row in action_values:
        key = (row["state_id"], row["action_id"])
        row["initial_policy_probability"] = initial_policy_probabilities[key]
        row["final_policy_probability"] = current_policy_probabilities[key]
    value_comparison = [{
        "state_id": row["state_id"], "gap_m": row["gap_m"],
        "speed_mps": row["speed_mps"], "is_terminal": row["is_terminal"],
        "initial_value": initial_values[row["state_id"]], "final_value": row["value"],
        "value_gain": row["value"] - initial_values[row["state_id"]],
    } for row in final_evaluation["values"]]
    return {
        "summary": {
            "converged": True, "policy_stable": True,
            "improvement_rounds": improvement_round, "evaluated_policies": len(history),
            "initial_state_id": initial_evaluation["summary"]["initial_state_id"],
            "initial_state_value_before": initial_evaluation["summary"]["initial_state_value"],
            "initial_state_value_after": final_evaluation["summary"]["initial_state_value"],
            "discount": discount, "action_value_tie_tolerance": action_value_tie_tolerance,
            "state_value_convergence_tolerance": state_value_convergence_tolerance,
            "optimality_residual": optimality_residual,
            "optimality_tolerance": optimality_tolerance,
            "optimal_value_error_bound": optimality_residual / (1 - discount),
        },
        "history": history, "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation, "action_values": action_values,
        "value_comparison": value_comparison,
    }
