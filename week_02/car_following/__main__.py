"""Run from week_02: python -m car_following init --database output/mdp.sqlite3."""

import argparse
import json
from pathlib import Path

from .database import initialize_database, inspect_database
from .evaluation import evaluate_policy
from .improvement import improve_policy, iterate_policy


def main():
    parser = argparse.ArgumentParser(description="Initialize, inspect, evaluate, improve, iterate, or simulate the car-following policy.")
    parser.add_argument("command", choices=("init", "inspect", "evaluate", "improve", "iterate", "simulate"))
    parser.add_argument("--database", type=Path, default=Path("output/car_following.sqlite3"))
    parser.add_argument("--policy", choices=("initial", "final"), default="final",
                        help="Policy to run with simulate (default: final).")
    parser.add_argument("--seed", type=int, default=0, help="Action-sampling seed for simulate.")
    args = parser.parse_args()
    if args.command == "simulate":
        from .simulation import run_simulation
        result = run_simulation(args.database, policy=args.policy, seed=args.seed)
        print(json.dumps({"summary": result["summary"], "trajectory": result["trajectory"]}, indent=2))
        return
    if args.command == "iterate":
        result = iterate_policy(args.database)
        print(json.dumps({"summary": result["summary"], "history": result["history"]}, indent=2))
        return
    if args.command == "improve":
        print(json.dumps(improve_policy(args.database)["summary"], indent=2))
        return
    if args.command == "evaluate":
        print(json.dumps(evaluate_policy(args.database)["summary"], indent=2))
        return
    if args.command == "init":
        initialize_database(args.database)
    print(json.dumps(inspect_database(args.database), indent=2))


if __name__ == "__main__":
    main()
