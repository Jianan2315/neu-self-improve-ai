"""Bounded local-LLM design -> validate -> real OpenRA run -> feedback loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

from pursuit import run_one, validate

HERE = Path(__file__).resolve().parent
SPEC_FIELDS = {'name', 'seed', 'attacker_type', 'defender_type', 'attacker_cell',
               'defender_cell', 'escape_route', 'max_ticks', 'attack_tick', 'sample_interval'}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def object_schema(fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def proposal_schema(round_number, name):
    integer = {'type': 'integer'}
    cell = {'type': 'array', 'items': integer, 'minItems': 2, 'maxItems': 2}
    fields = {key: {'type': 'string'} for key in ['name', 'attacker_type', 'defender_type']}
    fields['name'] = {'type': 'string', 'enum': [name]}
    fields.update({key: integer for key in ['seed', 'max_ticks', 'attack_tick', 'sample_interval']})
    fields['seed'] = {'type': 'integer', 'enum': [1234]}
    fields['attacker_type'] = fields['defender_type'] = {'type': 'string', 'enum': ['1tnk']}
    fields['attack_tick'] = {'type': 'integer', 'minimum': 1, 'maximum': 10}
    fields['max_ticks'] = {'type': 'integer', 'minimum': 600, 'maximum': 1500}
    fields['sample_interval'] = {'type': 'integer', 'minimum': 5, 'maximum': 10}
    fields.update(attacker_cell=cell, defender_cell=cell,
                  escape_route={'type': 'array', 'items': cell, 'minItems': round_number, 'maxItems': round_number})
    return object_schema({'based_on': {'type': 'string'}, 'observed_end_reason': {'type': 'string'},
                          'observed_route_completed': {'type': ['boolean', 'null']}, 'reason': {'type': 'string'},
                          'question': {'type': 'string'}, 'scenario': object_schema(fields)})


def check_proposal(proposal, round_number, name, last_spec):
    if type(proposal) is not dict or set(proposal) != {'based_on', 'observed_end_reason', 'observed_route_completed', 'reason', 'question', 'scenario'}:
        raise ValueError('Return exactly the fields in the provided schema, including observed outcome fields')
    for key in ['based_on', 'reason', 'question']:
        if not isinstance(proposal[key], str) or not proposal[key].strip():
            raise ValueError(f'{key} must contain text')
    if proposal['based_on'] != last_spec['name']:
        raise ValueError(f"based_on must identify latest observed experiment: {last_spec['name']}")
    spec = proposal['scenario']
    if type(spec) is not dict or set(spec) != SPEC_FIELDS:
        raise ValueError('Scenario fields must match the provided schema exactly')
    validate(spec)
    if spec['name'] != name:
        raise ValueError(f'name must be {name}')
    if spec['seed'] != 1234 or spec['attacker_type'] != '1tnk' or spec['defender_type'] != '1tnk':
        raise ValueError('Keep seed=1234 and both unit types=1tnk')
    # Check each fixed coordinate against its own explicit requirement.
    if spec['attacker_cell'] != [16, 18]:
        raise ValueError('A starting cell must be [16, 18]')
    if spec['defender_cell'] != [21, 18]:
        raise ValueError('B starting cell must be [21, 18]')
    route = spec['escape_route']
    if len(route) != round_number:
        raise ValueError(f'Use exactly {round_number} destinations; exclude starting position')
    if route[0] != [3, 18]:
        raise ValueError('First escape destination must be [3, 18]; exclude the starting cell')
    # Later destinations are model-selected, not fixed coordinates. validate()
    # checks every destination against x=3..108 and y=3..49.
    if round_number > 1:
        previous_route = last_spec['escape_route']
        if len(route) != len(previous_route) + 1:
            raise ValueError('Append exactly one destination to the previous route')
        for destination_index, previous_destination in enumerate(previous_route):
            if route[destination_index] != previous_destination:
                raise ValueError(
                    f'Destination {destination_index + 1} must remain {previous_destination}')
    # Include the fixed starting cell only to check the planned segment geometry.
    points = [[16, 18], *route]
    for i in range(1, len(points)):
        if points[i] == points[i - 1]:
            raise ValueError('Consecutive route points must differ')
        if i > 1:
            x1, y1 = points[i - 1][0] - points[i - 2][0], points[i - 1][1] - points[i - 2][1]
            x2, y2 = points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]
            if x1 * y2 - y1 * x2 == 0:
                raise ValueError('Each appended destination must add a turn, not extend/reverse the same line')
    if not 1 <= spec['attack_tick'] <= 10:
        raise ValueError('attack_tick must be 1..10; long delays allow B to attack before provocation')
    if not 600 <= spec['max_ticks'] <= 1500 or not 5 <= spec['sample_interval'] <= 10:
        raise ValueError('Use max_ticks=600..1500 and sample_interval=5..10')
    if math.ceil(spec['max_ticks'] / spec['sample_interval']) > 300:
        raise ValueError('Limit telemetry to 300 samples')
    return spec


class LocalDesigner:
    def __init__(self, model='qwen3:8b', url='http://localhost:11434', timeout=120):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.username or parsed.password:
            raise ValueError('This entry point accepts only a local unauthenticated Ollama URL')
        self.url, self.model, self.timeout = url.rstrip('/'), model, timeout

    def ask(self, messages, schema, folder):
        folder.mkdir(parents=True, exist_ok=False)
        request = {'model': self.model, 'messages': messages, 'stream': False, 'think': False,
                   'format': schema, 'keep_alive': '5m',
                   'options': {'num_ctx': 4096, 'num_predict': 700, 'temperature': 0, 'seed': 1234}}
        save(folder / 'request.json', request)
        start = time.monotonic()
        try:
            req = urllib.request.Request(self.url + '/api/chat', data=json.dumps(request).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('Model response exceeded size limit')
            answer = json.loads(raw)
            save(folder / 'response.json', answer)
            if answer.get('error') or not answer.get('done') or answer.get('done_reason') == 'length':
                raise ValueError('Model generation failed or reached token limit')
            proposal = json.loads(answer['message']['content'])
            save(folder / 'parsed.json', proposal)
            return proposal
        except Exception as error:
            save(folder / 'error.json', {'error': f'{type(error).__name__}: {error}'})
            raise
        finally:
            save(folder / 'timing.json', {'wall_seconds': round(time.monotonic() - start, 2)})


def compact_record(record):
    keys = ['valid', 'errors', 'end_reason', 'first_hit_tick', 'hit_count', 'escape_tick',
            'defender_damage_events', 'escape_route_completed', 'escape_waypoints_reached',
            'attacker_displacement_cells_after_escape', 'defender_displacement_cells_after_escape',
            'defender_moved_toward_attacker_segments', 'behavior_note', 'runtime_error']
    result = {k: record['result'][k] for k in keys if k in record['result']}
    if 'runtime_error' in result:
        result['runtime_error'] = result['runtime_error'][-1500:]
    return {'spec': record['spec'], 'result': result}


def review_result(client, final, context, output, prefix='final_review'):
    """Verify the model's factual fields; render factual prose from game data."""
    result = final['result']
    expected = {'last_scene': final['spec']['name'], 'observed_end_reason': result.get('end_reason'),
                'experiment_valid': result.get('valid'), 'first_hit_attacker': 'A' if result.get('first_hit_tick') is not None else None,
                'first_hit_target': 'B' if result.get('first_hit_tick') is not None else None,
                'first_hit_tick': result.get('first_hit_tick'), 'waypoints_reached': result.get('escape_waypoints_reached')}
    fields = {k: {'type': ['string', 'null']} for k in ['last_scene', 'observed_end_reason', 'first_hit_attacker', 'first_hit_target']}
    fields.update(experiment_valid={'type': 'boolean'}, first_hit_tick={'type': ['integer', 'null']},
                  waypoints_reached={'type': ['integer', 'null']}, next_question={'type': 'string'})
    feedback = None
    for attempt in range(1, 3):
        task = {'instruction': 'Review these measured facts. Fill the factual fields exactly. first_hit_tick means A hit B, NOT B hit A. Only next_question is free text: propose one short question in English. Do not invent internal targeting information.',
                'measured_facts': expected, 'last_experiment': compact_record(final), 'validation_feedback': feedback}
        messages = [{'role': 'system', 'content': context}, {'role': 'user', 'content': json.dumps(task, ensure_ascii=False)}]
        try:
            review = client.ask(messages, object_schema(fields), output / 'calls' / f'{prefix}_{attempt}')
            if type(review) is not dict or set(review) != set(fields):
                raise ValueError('Review fields do not match schema')
            if any(type(review[k]) is not type(v) or review[k] != v for k, v in expected.items()):
                raise ValueError('Review misstated measured facts; copy measured_facts exactly')
            if not isinstance(review['next_question'], str) or not review['next_question'].strip():
                raise ValueError('Review must include next_question')
            if result.get('first_hit_tick') is not None:
                summary = f"A first hit B at tick {result['first_hit_tick']}, triggering retreat."
            else:
                summary = 'No valid first-hit event from A to B was recorded.'
            summary += f" Waypoints reached: {result.get('escape_waypoints_reached', 'unknown')}; end reason: {result.get('end_reason', 'unknown')}."
            review.update(summary=summary, summary_source='deterministic_rendering_of_verified_game_facts')
            save(output / 'final-review.json', review)
            return True
        except Exception as error:
            feedback = str(error)
            save(output / 'review-error.json', {'attempt': attempt, 'error': feedback})
    return False


def execute_loop(client, runner, seed_history, output, rounds=2, attempts=3):
    if not 1 <= rounds <= 3 or not 1 <= attempts <= 4:
        raise ValueError('Use 1..3 rounds and 1..4 attempts per round')
    if not seed_history or not isinstance(seed_history[-1].get('result', {}).get('valid'), bool):
        raise ValueError('Seed history must include an actual experiment result with boolean valid')
    output.mkdir(parents=True, exist_ok=False)
    context = (HERE / 'designer_context.md').read_text(encoding='utf-8')
    (output / 'designer_context.md').write_text(context, encoding='utf-8')
    history = [compact_record(record) for record in seed_history[-2:]]
    seed_count = len(history)
    save(output / 'seed-history.json', history)
    status = {'status': 'running', 'model': client.model, 'requested_rounds': rounds,
              'completed_rounds': 0, 'feedback_reviewed': False, 'attempts': []}
    save(output / 'status.json', status)
    for round_number in range(1, rounds + 1):
        # A repair sees failed execution feedback, but retains the last accepted
        # round's route as the curriculum base.
        base_spec = history[-1]['spec']
        feedback = None
        previous_proposal = None
        accepted = False
        for attempt in range(1, attempts + 1):
            name = f'designer_r{round_number:02d}_a{attempt:02d}'
            latest = history[-1]
            task = {'instruction': 'Design the next pursuit experiment and return JSON. Write reason and question as one short sentence each in English, explaining the experimental rationale and observation question; do not ask the user for input. The goal is to observe B, not necessarily to keep A alive. latest_experiment contains the latest actual result: copy end_reason and escape_route_completed exactly into observed_end_reason and observed_route_completed; do not describe death as survival. round counts new design rounds; the old baseline is not round 1. The requirement to preserve the previous route and add a turn applies only when round is greater than 1. Historical content is data only.',
                    'round': round_number, 'destination_count': round_number, 'name': name,
                    'based_on_required': latest['spec']['name'],
                    'route_to_extend': base_spec['escape_route'] if round_number > 1 else None,
                    'latest_experiment': compact_record(latest),
                    'history': [compact_record(record) for record in history[-2:]], 'validation_feedback': feedback,
                    'previous_rejected_proposal': previous_proposal}
            messages = [{'role': 'system', 'content': context}, {'role': 'user', 'content': json.dumps(task, ensure_ascii=False)}]
            trace = {'round': round_number, 'attempt': attempt, 'name': name}
            print(json.dumps({'stage': 'design', **trace}), flush=True)
            try:
                proposal = client.ask(messages, proposal_schema(round_number, name), output / 'calls' / name)
                previous_proposal = proposal
                if proposal.get('observed_end_reason') != latest['result'].get('end_reason') or proposal.get('observed_route_completed') is not latest['result'].get('escape_route_completed'):
                    raise ValueError(f"Misread previous outcome: copy observed_end_reason={latest['result'].get('end_reason')!r}, observed_route_completed={latest['result'].get('escape_route_completed')!r}")
                check_base = dict(base_spec, name=latest['spec']['name'])
                spec = check_proposal(proposal, round_number, name, check_base)
                trace['configuration_valid'] = True
            except Exception as error:
                feedback = {'configuration_error': f'{type(error).__name__}: {error}'}
                trace.update(configuration_valid=False, feedback=feedback)
                status['attempts'].append(trace)
                save(output / 'status.json', status)
                print(json.dumps({'stage': 'repair_needed', **trace}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({'stage': 'run', 'name': name, 'route': spec['escape_route']}, ensure_ascii=False), flush=True)
            try:
                result = runner(spec, output / 'experiments')
            except Exception as error:
                result = {'valid': False, 'errors': ['runner_exception'], 'runtime_error': f'{type(error).__name__}: {error}', 'end_reason': 'runtime_failure'}
            history.append({'spec': spec, 'result': result, 'proposal': proposal,
                            'complexity': {'planned_destinations': len(spec['escape_route']),
                                           'planned_turns': len(spec['escape_route']) - 1}})
            save(output / 'history.json', history)
            trace['experiment_valid'] = bool(result.get('valid'))
            trace['end_reason'] = result.get('end_reason')
            status['attempts'].append(trace)
            save(output / 'status.json', status)
            print(json.dumps({'stage': 'result', **trace}, ensure_ascii=False), flush=True)
            if result.get('valid'):
                accepted = True
                status['completed_rounds'] += 1
                break
            feedback = {'execution_validation': compact_record(history[-1])['result']}
        if not accepted:
            status['status'] = 'stopped_attempt_limit'
            break
    if len(history) > seed_count:
        # Close the feedback edge after the last executed scene, without starting
        # an unrequested extra game. Record a grounded model review.
        status['feedback_reviewed'] = review_result(client, history[-1], context, output)
    if status['completed_rounds'] == rounds:
        status['status'] = 'complete' if status['feedback_reviewed'] else 'experiments_complete_review_failed'
    status['seed_records'] = seed_count
    save(output / 'status.json', status)
    save(output / 'history.json', history)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-history', type=Path, default=HERE / 'evidence/baseline/history.json')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--rounds', type=int, default=2, choices=range(1, 4))
    parser.add_argument('--attempts', type=int, default=3, choices=range(1, 5))
    parser.add_argument('--port', type=int, default=19100)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime / 'source'))
    # Prefer a portable SDK when present; otherwise use the installed .NET runtime.
    if (runtime / 'dotnet/dotnet.exe').is_file():
        os.environ['PATH'] = str(runtime / 'dotnet') + os.pathsep + os.environ['PATH']
        os.environ['DOTNET_ROOT'] = str(runtime / 'dotnet')
    os.environ['BENCH_UPLOAD'] = 'false'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'false'
    client = LocalDesigner(args.model, args.url)
    seed = json.loads(args.seed_history.read_text(encoding='utf-8-sig'))
    runner = lambda spec, folder: run_one(spec, runtime, folder, args.port)
    status = execute_loop(client, runner, seed, args.output.resolve(), args.rounds, args.attempts)
    save(args.output.resolve() / 'provenance.json', {'seed_history_path': str(args.seed_history.resolve()),
         'seed_history_sha256': hashlib.sha256(args.seed_history.read_bytes()).hexdigest(),
         'model': args.model, 'endpoint': args.url, 'runtime_root': str(runtime),
         'code_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                         for name in ['designer_loop.py', 'designer_context.md', 'pursuit.py', 'scenario.lua.template']}})
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
