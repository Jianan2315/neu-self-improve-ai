-- SQL is the schema source of truth. SQLAlchemy classes are reflected from it.
PRAGMA foreign_keys = ON;

CREATE TABLE environment (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL,
    specification_json TEXT NOT NULL,
    initial_state_id INTEGER NOT NULL REFERENCES state(id)
);

CREATE TABLE state (
    id INTEGER PRIMARY KEY,
    gap_m INTEGER,
    speed_mps INTEGER,
    is_terminal INTEGER NOT NULL CHECK (is_terminal IN (0, 1)),
    UNIQUE (gap_m, speed_mps),
    CHECK (
        (id = 0 AND is_terminal = 1 AND gap_m IS NULL AND speed_mps IS NULL)
        OR
        (id BETWEEN 1 AND 8000 AND is_terminal = 0
         AND gap_m IS NOT NULL AND speed_mps IS NOT NULL
         AND typeof(gap_m) = 'integer' AND typeof(speed_mps) = 'integer'
         AND gap_m BETWEEN 1 AND 200 AND speed_mps BETWEEN 1 AND 40
         AND id = (gap_m - 1) * 40 + speed_mps)
    )
);

CREATE TABLE action (
    id INTEGER PRIMARY KEY CHECK (id IN (0, 1, 2)),
    name TEXT NOT NULL UNIQUE,
    acceleration_mps2 INTEGER NOT NULL,
    CHECK ((id = 0 AND name = 'accelerate' AND acceleration_mps2 = 6)
        OR (id = 1 AND name = 'maintain' AND acceleration_mps2 = 0)
        OR (id = 2 AND name = 'decelerate' AND acceleration_mps2 = -6))
);

CREATE TABLE transition (
    state_id INTEGER NOT NULL REFERENCES state(id) CHECK (state_id <> 0),
    action_id INTEGER NOT NULL REFERENCES action(id),
    next_state_id INTEGER NOT NULL REFERENCES state(id),
    transition_probability REAL NOT NULL CHECK (transition_probability = 1.0),
    reward REAL NOT NULL,
    terminal_reason TEXT,
    elapsed_seconds REAL NOT NULL CHECK (elapsed_seconds BETWEEN 0 AND 1),
    gap_after_m REAL NOT NULL,
    speed_after_mps REAL NOT NULL,
    PRIMARY KEY (state_id, action_id),
    CHECK (
        (next_state_id <> 0 AND terminal_reason IS NULL AND elapsed_seconds = 1
         AND reward IN (-1, 0, 1))
        OR
        (next_state_id = 0 AND terminal_reason IS NOT NULL AND (
            (terminal_reason = 'collision' AND reward = -100)
            OR (terminal_reason = 'lost' AND reward = -50)
            OR (terminal_reason = 'stopped' AND reward = -20)
            OR (terminal_reason = 'overspeed' AND reward = -10)))
    )
);

CREATE INDEX transition_next_state ON transition(next_state_id);

CREATE TABLE policy (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE policy_probability (
    policy_id INTEGER NOT NULL REFERENCES policy(id),
    state_id INTEGER NOT NULL REFERENCES state(id) CHECK (state_id <> 0),
    action_id INTEGER NOT NULL REFERENCES action(id),
    policy_probability REAL NOT NULL CHECK (policy_probability BETWEEN 0 AND 1),
    PRIMARY KEY (policy_id, state_id, action_id),
    FOREIGN KEY (state_id, action_id) REFERENCES transition(state_id, action_id)
);
