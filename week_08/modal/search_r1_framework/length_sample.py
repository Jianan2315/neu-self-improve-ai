"""Test-only random 20-question natural search rollout using native veRL evaluation."""
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
from search_r1_framework.config import Settings, overrides, write_support_configs


def prepare_sample(prepared, output, manifest):
    import pandas as pd
    from transformers import AutoTokenizer
    from search_r1_framework.config import tool_config
    settings = replace(Settings(), prompt_tokens=384, micro_batch_per_gpu=16)
    frame = pd.read_parquet(prepared / 'train.parquet')
    if len(frame) != 2000:
        raise ValueError('Expected the existing 2000-question training set.')
    subset = frame.sample(n=20, random_state=42).copy()
    ids = [str(x['question_id']) for x in subset['extra_info']]
    if len(set(ids)) != 20:
        raise ValueError('The random sample must contain 20 different questions.')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    schemas = [x['tool_schema'] for x in tool_config()['tools']]
    prompt_lengths = []
    for raw in subset['prompt']:
        text = tokenizer.apply_chat_template(list(raw), tools=schemas, tokenize=False, add_generation_prompt=True)
        prompt_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
    if max(prompt_lengths) > settings.prompt_tokens:
        raise ValueError('An actual sampled prompt exceeds 384 tokens; do not truncate it.')
    subset.to_parquet(output / 'sample.parquet', index=False)
    selection = {'seed':42, 'source':'prepared train.parquet', 'source_questions':len(frame),
                 'sample_questions':20, 'responses_per_question':1, 'question_ids':ids,
                 'source_indices':[int(i) for i in subset.index], 'initial_prompt_tokens':prompt_lengths,
                 'sampling_temperature':1.0, 'inference_only':True}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    write_support_configs(output / 'runtime_config')
    path = output / 'runtime_config/agents.yaml'
    agents = json.loads(path.read_text())
    for agent in agents:
        if agent['name'] == 'assignment_search':
            agent['_target_'] = 'search_r1_framework.length_sample_agent.LengthSampleAgent'
    path.write_text(json.dumps(agents, indent=2))
    args = overrides(settings, 'grpo', prepared, output, manifest['model_path'],
                     str(prepared / 'initial_actor'), evaluation=True)
    values = {k:json.loads(v) for k,v in (a.split('=',1) for a in args)}
    values.update({'data.val_files':str(output/'sample.parquet'), 'data.val_batch_size':16,
                   'actor_rollout_ref.rollout.val_kwargs.n':1,
                   'actor_rollout_ref.rollout.val_kwargs.temperature':1.0,
                   'actor_rollout_ref.rollout.val_kwargs.do_sample':True})
    return settings, selection, [f'{k}={json.dumps(v)}' for k,v in values.items()]


def summarize_sample(output, selection):
    files = list((output/'evaluation').glob('*.jsonl'))
    if len(files) != 1:
        raise ValueError('Expected exactly one completed native evaluation export.')
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    ids = [str(x['question_id']) for x in rows]
    if len(rows) != 20 or len(set(ids)) != 20 or set(ids) != set(selection['question_ids']):
        raise ValueError('The evaluation export omitted or repeated selected questions.')
    stats = {}
    for key in ('initial_prompt_tokens','trajectory_tokens','model_tokens','tool_tokens','final_answer_tokens'):
        values = sorted(int(x[key]) for x in rows)
        stats[key] = {'min':min(values), 'mean':statistics.mean(values), 'p95':values[math.ceil(.95*len(values))-1], 'max':max(values)}
    return {'lengths':stats, 'response_limit_hits':sum(int(x['at_response_limit']) for x in rows),
            'answers_present':sum(int(x['has_answer']) for x in rows),
            'records':[{'question_id':x['question_id'], **{k:x[k] for k in (
                'initial_prompt_tokens','trajectory_tokens','model_tokens','tool_tokens','final_answer_tokens',
                'at_response_limit','has_answer','turn_count')}} for x in rows],
            'note':'One stochastic rollout per question from the initial policy. This is not a maximum over all training questions, later policies, or repeated samples; GPU memory is generation-only, not backward-pass capacity.'}
