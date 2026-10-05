# Week 2: Car-following MDP

A deterministic car-following MDP with a SQL database, SQLAlchemy ORM mappings,
policy evaluation, greedy policy improvement, and policy iteration.

See [explore.ipynb](explore.ipynb) for the data exploration, experiment results,
trajectory comparisons, and vehicle animations.

## Dependencies

Python 3.12 or later. Run the commands below from `week_02`.

```powershell
python -m pip install SQLAlchemy==2.0.54 pandas highway-env==1.12.1
```

## Database initialization and inspection

Create the database and populate the environment, states, actions, transitions,
and initial policy; then inspect the generated data:

```powershell
python -m car_following init
python -m car_following inspect
```

## Policy evaluation

Evaluate the initial policy, which selects each action with probability 1/3:

```powershell
python -m car_following evaluate
```

## Policy improvement

Perform one greedy improvement and evaluate the resulting policy:

```powershell
python -m car_following improve
```

## Policy iteration

Repeat evaluation and greedy improvement until the policy is stable and the
optimality residual meets the numerical tolerance:

```powershell
python -m car_following iterate
```

## HighwayEnv demonstration

Run the initial or final policy. We set the demonstration length to **20 decisions**;
this is a chosen display limit, not a 20-step limit on the underlying task.
A failure ends the run earlier.

```powershell
python -m car_following simulate --policy initial --seed 0
python -m car_following simulate --policy final --seed 0
```

These commands print the trajectory. `--seed` controls action sampling.

## Task specification

| Item | Definition |
| --- | --- |
| Road | One straight lane, two vehicles, no lane changes or overtaking |
| Lead vehicle | Constant speed of 20 m/s |
| Initial state | Bumper-to-bumper gap 50 m; following speed 20 m/s |
| Nonterminal states | Integer gap 1..200 m and following speed 1..40 m/s |
| State count | 8,000 nonterminal states and one shared terminal state |
| Actions | Accelerate +6 m/s², maintain 0, decelerate −6 m/s² |
| Decision interval | One second of constant acceleration, ending earlier if failure occurs |
| Transitions | Deterministic for a given state and action |
| Initial policy | Each action has probability 1/3 |
| Discount | 0.9 |

## Rewards and termination

| Condition | Reward | Terminated? |
| --- | ---: | --- |
| Collision: gap ≤ 0 m | −100 | Yes |
| Lost lead vehicle: gap > 200 m | −50 | Yes |
| Stopped: following speed reaches 0 m/s | −20 | Yes |
| Overspeed: following speed > 40 m/s | −10 | Yes |
| No failure and 0 < next gap < 40 m | −1 | No |
| No failure and 40 ≤ next gap ≤ 80 m | +1 | No |
| No failure and 80 < next gap ≤ 200 m | 0 | No |

The vehicle parameters and reward priorities are personal design choices.

## SQL and ORM structure

[car_following/schema.sql](car_following/schema.sql) defines the tables.
[car_following/database.py](car_following/database.py) initializes SQLite and
uses SQLAlchemy Automap to generate ORM classes from the table definitions.

`transition_probability` denotes the environment's transition probability;
`policy_probability` denotes the policy's action-selection probability.

| Table | Contents |
| --- | --- |
| `environment` | Task parameters and initial-state reference |
| `state` | Gap/speed combinations and the shared terminal state |
| `action` | Action names and accelerations |
| `transition` | Successor states, rewards, and motion records |
| `policy` | Policy identity and description |
| `policy_probability` | Action probabilities for each state |

## Validation

Local checks cover database integrity, probability normalization, motion and
termination rules, policy convergence, and agreement between HighwayEnv vehicle
coordinates and the SQL transitions.

## HighwayEnv integration

[car_following/simulation.py](car_following/simulation.py) subclasses HighwayEnv's
environment and vehicle classes and uses its road and rendering components.

## Sources

- [Week 2 assignment](https://app.notion.com/p/Week-2-Policy-Improvement-3e0e55cb6aa780b69724fb6e85235f60)
- [Lecture notebook](https://modal.com/notebooks/neu-info5100-oak-spr-2025/main/nb-x7ZeXpl1LG7mPa8udZJJLv)
- [HighwayEnv vehicle source](https://highway-env.farama.org/_modules/highway_env/vehicle/kinematics/)
- [SQLAlchemy Automap](https://docs.sqlalchemy.org/en/20/orm/extensions/automap.html)
