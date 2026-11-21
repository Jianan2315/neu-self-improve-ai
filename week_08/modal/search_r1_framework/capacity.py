"""Explicit test-only full-length workload; never used by formal training."""
import json


def prepare_capacity_inputs(prepared, output, settings, probe):
    import pandas as pd
    count = settings.questions_per_step * settings.steps
    source = pd.read_parquet(prepared / "train.parquet")
    indices = (probe.fixed_question_indices * settings.steps
               if probe.fixed_question_indices is not None else list(range(count)))
    frame = source.iloc[indices].copy()
    if len(frame) != count:
        raise ValueError("Insufficient prepared rows for the capacity check.")
    info = []
    for i, row in enumerate(frame["extra_info"]):
        info.append({**dict(row), "capacity_stress": i >= settings.questions_per_step * probe.warmup_steps})
    frame["extra_info"] = info
    frame.to_parquet(output / "capacity_train.parquet", index=False)
    validation = (source.iloc[probe.fixed_question_indices].copy() if probe.fixed_question_indices is not None
                  else pd.read_parquet(prepared / "dev.parquet").iloc[:16].copy())
    if probe.fixed_question_indices is None and len(validation) != 16:
        raise ValueError("Capacity validation requires one complete 16-question batch.")
    (output / 'question_selection.json').write_text(json.dumps({
        'training_row_indices': indices,
        'questions': [{'row_index': i, 'extra_info': dict(source.iloc[i]['extra_info']),
                       'prompt': str(source.iloc[i]['prompt'])} for i in dict.fromkeys(indices)]
    }, indent=2, default=str))
    validation["extra_info"] = [{**dict(x), "capacity_stress": True} for x in validation["extra_info"]]
    validation.to_parquet(output / "capacity_val.parquet", index=False)
    path = output / "runtime_config/agents.yaml"
    agents = json.loads(path.read_text())
    for agent in agents:
        if agent["name"] == "assignment_search":
            agent["_target_"] = "search_r1_framework.capacity_agent.CapacitySearchAgent"
    path.write_text(json.dumps(agents, indent=2))


def full_length_sampling(parameters, response_length):
    # veRL 0.6.1 supplies max_tokens from remaining context itself. Do not duplicate it.
    return {**parameters, "min_tokens": response_length, "ignore_eos": True,
            "stop": [], "stop_token_ids": []}


def extend_prompt(ids, target_length, fill_token):
    if len(ids) > target_length:
        raise ValueError("Stress prompt already exceeds configured prompt length.")
    return [fill_token] * (target_length - len(ids)) + list(ids)
