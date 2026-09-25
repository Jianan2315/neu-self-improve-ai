"""Generate, run and evaluate stock OpenRA pursuit scenarios without an LLM."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import threading
import traceback
import zipfile

HERE = Path(__file__).resolve().parent


def validate(spec):
    for key in ('seed', 'max_ticks', 'attack_tick', 'sample_interval'):
        if type(spec[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', spec['name']):
        raise ValueError('name must be a short lowercase identifier')
    # Deliberately restrict first prototype to stock ground combat units.
    allowed = {'arty', '1tnk', '2tnk', '3tnk', 'jeep', 'e1', 'e3'}
    for key in ('attacker_type', 'defender_type'):
        if spec[key] not in allowed:
            raise ValueError(f'{key} must be one of {sorted(allowed)}')
    if not spec['escape_route'] or len(spec['escape_route']) > 5:
        raise ValueError('escape_route must contain 1 to 5 cells')
    for cell in [spec['attacker_cell'], spec['defender_cell'], *spec['escape_route']]:
        if len(cell) != 2 or any(type(v) is not int for v in cell):
            raise ValueError('Each cell must contain two integers')
        if not (3 <= cell[0] <= 108 and 3 <= cell[1] <= 49):
            raise ValueError(f'Cell outside supported map bounds: {cell}')
    if spec['attacker_cell'] == spec['defender_cell']:
        raise ValueError('Units cannot start in the same cell')
    if not 1 <= spec['sample_interval'] <= 25:
        raise ValueError('sample_interval must be 1..25')
    if not 1 <= spec['attack_tick'] < spec['max_ticks'] <= 2000:
        raise ValueError('Require 1 <= attack_tick < max_ticks <= 2000')
    if math.ceil(spec['max_ticks'] / spec['sample_interval']) > 300:
        raise ValueError('Keep at most 300 telemetry samples per run in this prototype')


def generate(spec, engine, folder):
    validate(spec)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(engine / 'mods/ra/maps/singles.oramap') as source:
        original = source.read('map.yaml').decode()
        binary = bytearray(source.read('map.bin'))
    # Preserve the stock map format but make its terrain uniformly traversable.
    # Copy a known traversable terrain tile at (12,18); map storage is column-major.
    version, width, height = struct.unpack_from('<BHH', binary)
    if (width, height) != (112, 54):
        raise ValueError('This prototype expects the checked Singles map dimensions')
    if version == 1:
        tiles, heights, resources = 5, 0, 5 + width * height * 3
    elif version == 2:
        tiles, heights, resources = struct.unpack_from('<III', binary, 5)
    else:
        raise ValueError('Unsupported map binary format')
    pos = tiles + (12 * height + 18) * 3
    tile = binary[pos:pos + 3]
    binary[tiles:tiles + width * height * 3] = tile * (width * height)
    if resources:
        binary[resources:resources + width * height * 2] = bytes(width * height * 2)
    if heights:
        binary[heights:heights + width * height] = bytes(width * height)
    header = original.split('\nActors:', 1)[0]
    header = re.sub(r'^Title:.*$', 'Title: ' + spec['name'], header, flags=re.M)
    header = re.sub(r'^Author:.*$', 'Author: Pursuit prototype (terrain format from Singles)', header, flags=re.M)
    a, b = spec['attacker_cell'], spec['defender_cell']
    actors = f'''\nActors:
\tSpawn0: mpspawn
\t\tOwner: Neutral
\t\tLocation: 12,16
\tSpawn1: mpspawn
\t\tOwner: Neutral
\t\tLocation: 95,11
\tPursuitA: {spec['attacker_type']}
\t\tOwner: Multi1
\t\tLocation: {a[0]},{a[1]}
\t\tStance: HoldFire
\tPursuitB: {spec['defender_type']}
\t\tOwner: Multi0
\t\tLocation: {b[0]},{b[1]}
\t\tStance: AttackAnything

Rules: pursuit-rules.yaml
'''
    rules = '''World:
\t-SpawnStartingUnits:
\tLuaScript:
\t\tScripts: pursuit.lua
Player:
\t-ConquestVictoryConditions:
'''
    lua = (HERE / 'scenario.lua.template').read_text(encoding='utf-8')
    replacements = {'__NAME__': spec['name'], '__ATTACK_TICK__': str(spec['attack_tick']),
                    '__MAX_TICKS__': str(spec['max_ticks']), '__SAMPLE_INTERVAL__': str(spec['sample_interval']),
                    '__ROUTE_CELLS__': ', '.join(f'CPos.New({x}, {y})' for x, y in spec['escape_route']),
                    '__ESCAPE_COMMANDS__': '\n                '.join(
                        f'PursuitA.Move(CPos.New({x}, {y}))' for x, y in spec['escape_route'])}
    for key, value in replacements.items():
        lua = lua.replace(key, value)
    files = {'map.yaml': (header + actors).encode(), 'map.bin': bytes(binary),
             'pursuit-rules.yaml': rules.encode(), 'pursuit.lua': lua.encode()}
    path = folder / (spec['name'] + '.oramap')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, value)
    for name in ('map.yaml', 'pursuit-rules.yaml', 'pursuit.lua'):
        (folder / name).write_bytes(files[name])
    (folder / 'spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    return path


def parse_events(text, name):
    events = []
    for line in text.splitlines():
        if 'PURSUIT ' not in line:
            continue
        event = json.loads(line.split('PURSUIT ', 1)[1])
        if event['scenario'] == name:
            events.append(event)
    return events


def evaluate(events, spec):
    hits = [e for e in events if e['kind'] == 'a_hit_b']
    escapes = [e for e in events if e['kind'] == 'escape_command']
    ends = [e for e in events if e['kind'] == 'finished']
    samples = [e for e in events if e['kind'] == 'sample']
    retreat_tick = escapes[0]['tick'] if escapes else None
    after = [e for e in samples if retreat_tick is not None and e['tick'] >= retreat_tick]
    live = [e for e in after if e['a']['alive'] and e['b']['alive']]
    origin = next((e for e in samples if retreat_tick is not None and e['tick'] == retreat_tick), None)
    b0 = [origin['b']['x'], origin['b']['y']] if origin and origin['b']['alive'] else spec['defender_cell']
    max_b_move = max((math.hypot(e['b']['x'] - b0[0], e['b']['y'] - b0[1]) for e in live), default=0)
    a0 = [origin['a']['x'], origin['a']['y']] if origin and origin['a']['alive'] else spec['attacker_cell']
    max_a_move = max((math.hypot(e['a']['x'] - a0[0], e['a']['y'] - a0[1]) for e in live), default=0)
    # A geometric observation, not proof of an internal aggro target or its cause.
    following_segments = 0
    for prev, cur in zip(live, live[1:]):
        dx, dy = cur['b']['wx'] - prev['b']['wx'], cur['b']['wy'] - prev['b']['wy']
        tx, ty = prev['a']['wx'] - prev['b']['wx'], prev['a']['wy'] - prev['b']['wy']
        if dx * tx + dy * ty > 0:
            following_segments += 1
    errors = []
    if not any(e['kind'] == 'loaded' for e in events): errors.append('missing_loaded_event')
    if len(hits) != 1: errors.append(f'expected_one_damage_event_got_{len(hits)}')
    if len(escapes) != 1: errors.append('missing_or_duplicate_escape_command')
    if not samples: errors.append('missing_telemetry')
    if not ends: errors.append('missing_end_event')
    if escapes and hits and escapes[0]['tick'] != hits[0]['tick']: errors.append('escape_not_at_first_hit')
    if hits and any(e['kind'] == 'b_hit_a' and e['tick'] < hits[0]['tick'] for e in events):
        errors.append('defender_damaged_attacker_before_provocation')
    return {'valid': not errors, 'errors': errors, 'hit_count': len(hits),
            'first_hit_tick': hits[0]['tick'] if hits else None,
            'escape_tick': retreat_tick, 'end_reason': ends[-1]['reason'] if ends else 'runner_limit',
            'attacker_displacement_cells_after_escape': round(max_a_move, 3),
            'defender_displacement_cells_after_escape': round(max_b_move, 3),
            'defender_moved_toward_attacker_segments': following_segments,
            'defender_damage_events': sum(e['kind'] == 'b_hit_a' for e in events),
            'escape_waypoints_reached': sum(e['kind'] == 'waypoint_reached' for e in events),
            'escape_route_completed': sum(e['kind'] == 'waypoint_reached' for e in events) == len(spec['escape_route']),
            'behavior_note': 'Motion is measured from scene telemetry; internal target lock and disengagement cause are not exposed.'}


def run_one(spec, runtime, output, port):
    from openra_env.config import OpenRARLConfig
    from openra_env.server.openra_environment import OpenRAEnvironment
    from openra_env.server.openra_process import OpenRAConfig, OpenRAProcessManager
    import grpc
    from openra_env.server.bridge_client import commands_to_proto
    class LineLoggedProcess(OpenRAProcessManager):
        """Capture complete lines immediately, including when engine file logs buffer."""
        def _start_log_drainers(self):
            self.readers = []
            for name in ('stdout', 'stderr'):
                stream = getattr(self._process, name)
                reader = threading.Thread(target=self._read_lines, args=(stream, name), daemon=True)
                reader.start()
                self.readers.append(reader)

        def _read_lines(self, stream, name):
            with stream:
                for line in iter(stream.readline, b''):
                    self._append_log(name, line.decode('utf-8', errors='replace'))

    folder = output / spec['name']
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {folder}')
    engine = runtime / 'source/OpenRA'
    map_path = generate(spec, engine, folder)
    support = folder / 'engine-support'
    support.mkdir()
    cfg = OpenRARLConfig()
    cfg.game.openra_path = str(engine)
    cfg.game.grpc_port = port
    cfg.game.headless = True
    cfg.game.record_replays = False
    cfg.opponent.bot_type = 'dummy'
    cfg.planning.enabled = False
    cfg.agent.bench_upload = False
    daemon = LineLoggedProcess(OpenRAConfig(openra_path=str(engine), grpc_port=port,
        multi_session=True, extra_args={'Engine.SupportDir': str(support)}))
    env = None
    failure = None
    max_seen_tick = 0
    controller_actions = []
    pending_commands = None
    native_stop_sent = False
    try:
        daemon.launch()
        channel = grpc.insecure_channel(f'127.0.0.1:{port}')
        try:
            grpc.channel_ready_future(channel).result(timeout=30)
        finally:
            channel.close()
        env = OpenRAEnvironment(config=cfg, multi_session=True)
        env.reset(seed=spec['seed'], map_name='_scenario_' + spec['name'] + '.oramap',
                  map_data=base64.b64encode(map_path.read_bytes()).decode())
        # Lua triggers retreat. Unlike Lua Stop (CancelActivity only), the native
        # Stop order also clears AttackFollow's persistent turret target. Send it
        # once and requeue A's route; never send any command to B.
        with (folder / 'observations.jsonl').open('w', encoding='utf-8') as obs_file:
            from google.protobuf.json_format import MessageToDict
            deadline = time.monotonic() + 90
            while max_seen_tick < spec['max_ticks'] + 5:
                if time.monotonic() > deadline:
                    raise TimeoutError('Scenario exceeded 90 seconds wall time')
                obs = env._bridge.fast_advance_unary(5, commands=pending_commands)
                if pending_commands:
                    controller_actions[-1]['observed_after_tick'] = obs.tick
                    pending_commands = None
                max_seen_tick = obs.tick
                observation = MessageToDict(obs, preserving_proto_field_name=True)
                observation.pop('spatial_map', None)  # Repeated full-map pixels are not needed for these measurements.
                obs_file.write(json.dumps(observation) + '\n')
                text = daemon.get_stdout()
                # The engine may still be writing the last JSON line. Do not parse
                # an actively written log as a completed JSONL file.
                if '"kind":"finished"' in text or obs.done:
                    break
                if not native_stop_sent and '"kind":"escape_command"' in text:
                    own = [u for u in obs.units if u.owner == 'Multi1' and u.type == spec['attacker_type']]
                    if len(own) != 1:
                        raise ValueError('Could not uniquely identify A for native Stop order')
                    commands = [{'action': 'stop', 'actor_id': own[0].actor_id}]
                    commands.extend({'action': 'move', 'actor_id': own[0].actor_id,
                                     'target_x': x, 'target_y': y, 'queued': True}
                                    for x, y in spec['escape_route'])
                    pending_commands = commands_to_proto(commands).commands
                    controller_actions.append({'issued_after_tick': obs.tick, 'commands': commands})
                    native_stop_sent = True
    except Exception:
        failure = traceback.format_exc()
    finally:
        try:
            if env is not None: env.close()
        finally:
            daemon.kill()
            for reader in getattr(daemon, 'readers', []):
                reader.join(timeout=2)
        (folder / 'engine-stdout.log').write_text(daemon.get_stdout(), encoding='utf-8')
        (folder / 'engine-stderr.log').write_text(daemon.get_stderr(), encoding='utf-8')
    # stdout is drained continuously by the process manager; the engine's file
    # logger buffers data and can leave a partial last line when terminated.
    text = daemon.get_stdout()
    try:
        events = parse_events(text, spec['name'])
    except (ValueError, KeyError):
        events = []
        failure = (failure or '') + '\nIncomplete or invalid final telemetry:\n' + traceback.format_exc()
    (folder / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    fields = ['tick', 'role', 'alive', 'x', 'y', 'wx', 'wy', 'hp', 'max_hp', 'idle', 'stance']
    with (folder / 'trajectory.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for event in events:
            if event['kind'] == 'sample':
                for role in ('a', 'b'):
                    writer.writerow({'tick': event['tick'], 'role': role, **event[role]})
    result = evaluate(events, spec)
    (folder / 'controller-actions.json').write_text(json.dumps(controller_actions, indent=2), encoding='utf-8')
    result['native_stop_after_tick'] = controller_actions[0]['issued_after_tick'] if controller_actions else None
    if result['escape_tick'] is not None and not any('observed_after_tick' in a for a in controller_actions):
        result['valid'] = False
        result['errors'].append('native_stop_not_confirmed_by_advance')
    result.update(name=spec['name'], max_seen_tick=max_seen_tick,
                  process_stopped=daemon.pid is None, map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest())
    if failure:
        result.update(valid=False, runtime_error=failure)
    if 'Fatal Lua Error' in text:
        result['valid'] = False
        result['errors'].append('fatal_lua_error')
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=HERE / 'scenario.json')
    parser.add_argument('--output', type=Path, default=HERE / 'runs')
    parser.add_argument('--rounds', type=int, default=1, choices=range(1, 4))
    parser.add_argument('--port', type=int, default=19100)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(runtime / 'source'))
    # Prefer a portable SDK when present; otherwise use the installed .NET runtime.
    if (runtime / 'dotnet/dotnet.exe').is_file():
        os.environ['PATH'] = str(runtime / 'dotnet') + os.pathsep + os.environ['PATH']
        os.environ['DOTNET_ROOT'] = str(runtime / 'dotnet')
    os.environ['BENCH_UPLOAD'] = 'false'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'false'
    spec = json.loads(args.spec.read_text(encoding='utf-8-sig'))
    validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for i in range(args.rounds):
        current = json.loads(json.dumps(spec))
        if i:
            # Explicit fixed curriculum, gated on valid execution, not on pursuit success.
            current['name'] = spec['name'] + f'_r{i + 1}'
            if i == 1:
                current['escape_route'] = [[8, 18], [8, 28]]
            else:
                current['escape_route'] = [[8, 18], [8, 28], [24, 28]]
        result = run_one(current, runtime, output, args.port)
        decision = ('stop_and_inspect' if not result['valid'] else
                    'increase_route_waypoints' if result['escape_route_completed'] else
                    'stop_for_review_base_route_not_completed')
        history.append({'spec': current, 'result': result, 'next_decision': decision})
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result['valid']:
            return 1
        if decision != 'increase_route_waypoints':
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
