"""Build a new SQLite database, then generate ORM mappings from its SQL schema."""

import json
import sqlite3
from pathlib import Path

from sqlalchemy import URL, create_engine, event
from sqlalchemy.ext.automap import automap_base

from .mdp import ACTIONS, SPEC, state_id, transition


def open_engine(path: Path):
    engine = create_engine(URL.create("sqlite", database=str(path.resolve())))

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    return engine


def reflected_models(engine):
    """Regenerate SQLAlchemy ORM classes without duplicating SQL definitions."""
    base = automap_base()
    # Explicit foreign-key columns suffice here; suppress ambiguous automatic
    # relationships between transition and its two references to state.
    base.prepare(autoload_with=engine, generate_relationship=lambda *args, **kwargs: None)
    return base


def initialize_database(path: Path) -> None:
    """Create once; never replace an existing database or experiment."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.executescript(Path("car_following/schema.sql").read_text(encoding="utf-8"))
        with connection:
            connection.execute("INSERT INTO state VALUES (0, NULL, NULL, 1)")
            states = [
                (state_id(gap, speed), gap, speed, 0)
                for gap in range(1, SPEC.max_gap_m + 1)
                for speed in range(1, SPEC.max_speed_mps + 1)
            ]
            connection.executemany("INSERT INTO state VALUES (?, ?, ?, ?)", states)
            connection.execute(
                "INSERT INTO environment VALUES (1, ?, ?, ?)",
                ("Exact single-lane car following", json.dumps(SPEC.as_dict(), sort_keys=True),
                 state_id(SPEC.initial_gap_m, SPEC.initial_speed_mps)),
            )
            connection.executemany("INSERT INTO action VALUES (?, ?, ?)", ACTIONS)
            records = []
            for identifier, gap, speed, _ in states:
                for action_id, _, _ in ACTIONS:
                    result = transition(gap, speed, action_id)
                    records.append((identifier, action_id, result.next_state_id, 1.0,
                                    result.reward, result.reason, result.elapsed_seconds,
                                    result.gap_after_m, result.speed_after_mps))
            connection.executemany(
                "INSERT INTO transition (state_id, action_id, next_state_id, transition_probability, "
                "reward, terminal_reason, elapsed_seconds, gap_after_m, speed_after_mps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records,
            )
            connection.execute(
                "INSERT INTO policy VALUES (1, ?, ?)",
                ("uniform_initial", "Each action has probability 1/3 at each nonterminal state."),
            )
            connection.execute(
                "INSERT INTO policy_probability (policy_id, state_id, action_id, policy_probability) "
                "SELECT 1, state_id, action_id, 1.0 / 3.0 FROM transition"
            )
    finally:
        connection.close()


def inspect_database(path: Path) -> dict:
    """Verify structural invariants and query the initial transitions via ORM."""
    if not path.is_file():
        raise FileNotFoundError(path)
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    engine = open_engine(path)
    try:
        base = reflected_models(engine)
        with Session(engine) as session:
            State = base.classes.state
            Action = base.classes.action
            Transition = base.classes.transition
            PolicyProbability = base.classes.policy_probability
            counts = {
                "nonterminal_states": session.scalar(select(func.count()).select_from(State).where(State.is_terminal == 0)),
                "terminal_states": session.scalar(select(func.count()).select_from(State).where(State.is_terminal == 1)),
                "actions": session.scalar(select(func.count()).select_from(Action)),
                "transitions": session.scalar(select(func.count()).select_from(Transition)),
                "initial_policy_entries": session.scalar(select(func.count()).select_from(PolicyProbability).where(PolicyProbability.policy_id == 1)),
            }
            expected = {"nonterminal_states": 8000, "terminal_states": 1, "actions": 3,
                        "transitions": 24000, "initial_policy_entries": 24000}
            if counts != expected:
                raise ValueError(f"Unexpected database counts: {counts}")
            initial = session.execute(
                select(Action.name, Transition.gap_after_m, Transition.speed_after_mps,
                       Transition.reward, Transition.terminal_reason)
                .join(Action, Transition.action_id == Action.id)
                .where(Transition.state_id == state_id(SPEC.initial_gap_m, SPEC.initial_speed_mps))
                .order_by(Action.id)
            ).mappings().all()
            reasons = dict(session.execute(
                select(Transition.terminal_reason, func.count())
                .where(Transition.terminal_reason.is_not(None))
                .group_by(Transition.terminal_reason)
            ).all())

        with engine.connect() as connection:
            checks = {
                "foreign_keys": connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall(),
                "policy_sums": connection.exec_driver_sql(
                    "SELECT state_id FROM policy_probability WHERE policy_id = 1 "
                    "GROUP BY state_id HAVING COUNT(*) <> 3 OR ABS(SUM(policy_probability) - 1) > 1e-12 "
                    "OR MAX(ABS(policy_probability - 1.0/3.0)) > 1e-12"
                ).fetchall(),
                "next_state_coordinates": connection.exec_driver_sql(
                    "SELECT t.state_id, t.action_id FROM transition t "
                    "JOIN state s ON s.id = t.next_state_id WHERE s.is_terminal = 0 "
                    "AND (ABS(t.gap_after_m - s.gap_m) > 1e-10 "
                    "OR ABS(t.speed_after_mps - s.speed_mps) > 1e-10)"
                ).fetchall(),
            }
            if any(checks.values()):
                raise ValueError(f"Database validation failed: {checks}")
            if connection.exec_driver_sql("PRAGMA integrity_check").scalar_one() != "ok":
                raise ValueError("SQLite integrity check failed.")

        return {"counts": counts, "terminal_transition_counts": reasons,
                "initial_transitions": [dict(row) for row in initial],
                "checks": "passed", "orm": "SQLAlchemy Automap from schema.sql",
                "policy_evaluation_run": False, "highwayenv_run": False}
    finally:
        engine.dispose()
