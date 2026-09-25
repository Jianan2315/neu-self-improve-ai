"""Critical contracts for rejecting model output and propagating real feedback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from designer_loop import check_proposal, execute_loop, LocalDesigner, review_result

BASE = {'name': 'observed_seed', 'seed': 1234, 'attacker_type': 'arty', 'defender_type': '1tnk',
        'attacker_cell': [16, 18], 'defender_cell': [21, 18], 'escape_route': [[8, 18]],
        'max_ticks': 600, 'attack_tick': 10, 'sample_interval': 5}
SEED = [{'spec': BASE, 'result': {'valid': True, 'end_reason': 'attacker_dead'}}]


def proposal(name, prior, count=1):
    spec = dict(copy.deepcopy(BASE), name=name, attacker_type='1tnk',
                escape_route=[[3, 18]] if count == 1 else [[3, 18], [3, 30]])
    return {'based_on': prior, 'observed_end_reason': 'attacker_dead', 'observed_route_completed': None,
            'reason': 'Compare actual results', 'question': 'Observe route changes', 'scenario': spec}


class FakeClient:
    model = 'test-double'

    def __init__(self, reject_first=False, always_bad=False):
        self.calls = []
        self.reject_first = reject_first
        self.always_bad = always_bad

    def ask(self, messages, schema, folder):
        task = json.loads(messages[-1]['content'])
        self.calls.append(task)
        if 'measured_facts' in task:
            return dict(task['measured_facts'], next_question='What should the next experiment measure?')
        answer = proposal(task['name'], task['based_on_required'], task['round'])
        answer['observed_end_reason'] = task['history'][-1]['result']['end_reason']
        answer['observed_route_completed'] = task['history'][-1]['result'].get('escape_route_completed')
        if self.always_bad or (self.reject_first and len(self.calls) == 1):
            answer['scenario']['escape_route'] = [[108, 18]]
        return answer


class DesignerContracts(unittest.TestCase):
    def test_wrong_direction_and_unsafe_name_rejected(self):
        for mutate in [lambda p: p['scenario'].update(escape_route=[[108, 18]]),
                       lambda p: p['scenario'].update(name='../escape')]:
            p = proposal('designer_r01_a01', BASE['name'])
            mutate(p)
            with self.assertRaises(ValueError): check_proposal(p, 1, 'designer_r01_a01', BASE)

    def test_stale_history_reference_rejected(self):
        with self.assertRaises(ValueError):
            check_proposal(proposal('r1', 'invented'), 1, 'r1', BASE)

    def test_added_collinear_destination_is_not_a_turn(self):
        p = proposal('r2', 'r1', 2)
        p['scenario']['escape_route'][-1] = [8, 18]
        with self.assertRaises(ValueError):
            check_proposal(p, 2, 'r2', dict(BASE, name='r1', escape_route=[[3, 18]]))

    def test_invalid_output_never_runs_and_real_feedback_reaches_next_round(self):
        client = FakeClient(reject_first=True)
        ran = []
        def run(spec, folder):
            ran.append(spec)
            return {'valid': True, 'end_reason': 'time_limit', 'first_hit_tick': 37,
                    'escape_route_completed': False, 'escape_waypoints_reached': 0}
        with tempfile.TemporaryDirectory() as directory:
            status = execute_loop(client, run, SEED, Path(directory) / 'run', 2, 3)
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(ran), 2)
        self.assertIn('configuration_error', client.calls[1]['validation_feedback'])
        self.assertEqual(client.calls[2]['history'][-1]['result']['first_hit_tick'], 37)
        self.assertEqual(client.calls[-1]['last_experiment']['spec']['name'], ran[-1]['name'])

    def test_rejected_model_is_bounded(self):
        client = FakeClient(always_bad=True)
        def must_not_run(spec, folder):
            raise AssertionError('Rejected output reached runner')
        with tempfile.TemporaryDirectory() as directory:
            status = execute_loop(client, must_not_run, SEED, Path(directory) / 'run', 2, 2)
        self.assertEqual(status['status'], 'stopped_attempt_limit')
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(status['completed_rounds'], 0)

    def test_invalid_execution_is_returned_to_model_before_retry(self):
        client = FakeClient()
        ran = []
        def run(spec, folder):
            ran.append(spec)
            return {'valid': len(ran) > 1, 'end_reason': 'time_limit', 'errors': [] if len(ran) > 1 else ['missing_end_event']}
        with tempfile.TemporaryDirectory() as directory:
            status = execute_loop(client, run, SEED, Path(directory) / 'run', 1, 2)
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(client.calls[1]['validation_feedback']['execution_validation']['errors'], ['missing_end_event'])
        self.assertEqual(client.calls[1]['based_on_required'], ran[0]['name'])

    def test_no_remote_endpoint(self):
        with self.assertRaises(ValueError): LocalDesigner(url='https://example.com')

    def test_reversed_attack_roles_are_rejected(self):
        class WrongRoles(FakeClient):
            def ask(self, messages, schema, folder):
                answer = super().ask(messages, schema, folder)
                answer.update(first_hit_attacker='B', first_hit_target='A')
                return answer
        final = {'spec': BASE, 'result': {'valid': True, 'end_reason': 'attacker_dead', 'first_hit_tick': 30, 'escape_waypoints_reached': 1}}
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(review_result(WrongRoles(), final, '', Path(directory)))


if __name__ == '__main__': unittest.main()
