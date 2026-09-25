"""Contract checks: valid non-pursuit is allowed; missing evidence is not."""
import copy
import json
import unittest
from pathlib import Path
from pursuit import evaluate, validate, parse_events

SPEC = json.loads((Path(__file__).parent / 'scenario.json').read_text())


class EvaluationContract(unittest.TestCase):
    def events(self):
        state = {'alive': True, 'x': 16, 'y': 18, 'wx': 16384, 'wy': 18432}
        other = dict(state, x=21, wx=21504)
        return [dict(kind='loaded', tick=0), dict(kind='attack_command', tick=10),
                dict(kind='a_hit_b', tick=67), dict(kind='escape_command', tick=67),
                dict(kind='sample', tick=67, a=state, b=other),
                dict(kind='sample', tick=100, a=dict(state, x=15, wx=15360), b=other),
                dict(kind='finished', tick=600, reason='time_limit')]

    def test_no_pursuit_is_still_valid(self):
        result = evaluate(self.events(), SPEC)
        self.assertTrue(result['valid'])
        self.assertEqual(result['defender_moved_toward_attacker_segments'], 0)

    def test_missing_first_hit_is_invalid(self):
        events = [e for e in self.events() if e['kind'] != 'a_hit_b']
        self.assertFalse(evaluate(events, SPEC)['valid'])

    def test_multiple_damage_events_break_one_hit_condition(self):
        events = self.events() + [dict(kind='a_hit_b', tick=90)]
        self.assertFalse(evaluate(events, SPEC)['valid'])

    def test_early_defender_attack_is_reported(self):
        events = self.events() + [dict(kind='b_hit_a', tick=20)]
        self.assertIn('defender_damaged_attacker_before_provocation', evaluate(events, SPEC)['errors'])

    def test_scene_logs_are_filtered_by_name(self):
        line = 'Lua debug: PURSUIT {"scenario":"other","kind":"finished","tick":1}'
        self.assertEqual(parse_events(line, 'pursuit_001'), [])

    def test_overlapping_units_and_unsafe_names_rejected(self):
        spec = copy.deepcopy(SPEC)
        spec['defender_cell'] = spec['attacker_cell']
        with self.assertRaises(ValueError): validate(spec)
        spec = dict(SPEC, name='../outside')
        with self.assertRaises(ValueError): validate(spec)


if __name__ == '__main__':
    unittest.main()
