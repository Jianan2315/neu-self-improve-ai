"""Cheap regressions for the explicitly isolated length-boundary workload."""
import json
from pathlib import Path
import tempfile
import unittest
from search_r1_framework.capacity import full_length_sampling, extend_prompt
from search_r1_framework.config import Settings, overrides
from search_r1_framework.performance import PerformanceProbe, performance_overrides


class CapacityTests(unittest.TestCase):
    def test_fixed_two_questions_repeat_without_adding_validation_questions(self):
        import pandas as pd
        from search_r1_framework.capacity import prepare_capacity_inputs
        probe = PerformanceProbe(long_sequence_check=True, fixed_question_indices=[4, 5], measured_steps=2)
        settings = probe.settings_for(Settings())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, output = root / 'prepared', root / 'output'
            prepared.mkdir()
            (output / 'runtime_config').mkdir(parents=True)
            (output / 'runtime_config/agents.yaml').write_text(json.dumps([{'name': 'assignment_search'}]))
            pd.DataFrame({'extra_info': [{'question_id': str(i)} for i in range(8)],
                          'prompt': ['question ' + str(i) for i in range(8)]}).to_parquet(prepared / 'train.parquet')
            prepare_capacity_inputs(prepared, output, settings, probe)
            train = pd.read_parquet(output / 'capacity_train.parquet')
            validation = pd.read_parquet(output / 'capacity_val.parquet')
            self.assertEqual([x['question_id'] for x in train.extra_info], ['4', '5'] * 4)
            self.assertEqual([x['capacity_stress'] for x in train.extra_info], [False] * 4 + [True] * 4)
            self.assertEqual([x['question_id'] for x in validation.extra_info], ['4', '5'])
        with self.assertRaises(ValueError):
            PerformanceProbe(long_sequence_check=True, fixed_question_indices=[4, 4]).settings_for(Settings())

    def test_no_duplicate_max_tokens_or_sampling_mutation(self):
        source = {'temperature': 1, 'stop': ['end']}
        result = full_length_sampling(source, 4096)
        self.assertNotIn('max_tokens', result)
        self.assertEqual(result['min_tokens'], 4096)
        self.assertTrue(result['ignore_eos'])
        self.assertEqual(source['stop'], ['end'])

    def test_real_attention_tokens_not_padding(self):
        self.assertEqual(extend_prompt([1, 2], 5, 20), [20, 20, 20, 1, 2])
        with self.assertRaises(ValueError):
            extend_prompt([1, 2], 1, 20)

    def test_only_explicit_probe_changes_dataset_and_validation(self):
        args = overrides(Settings(), 'ppo', Path('/data/p'), Path('/results/r'), '/model', '/actor', '/critic')
        normal = dict(x.split('=', 1) for x in performance_overrides(args, PerformanceProbe()))
        stress = dict(x.split('=', 1) for x in performance_overrides(args, PerformanceProbe(long_sequence_check=True)))
        self.assertEqual(json.loads(normal['data.train_files']), '/data/p/train.parquet')
        self.assertEqual(json.loads(normal['trainer.test_freq']), -1)
        self.assertEqual(json.loads(stress['data.train_files']), '/results/r/capacity_train.parquet')
        self.assertFalse(json.loads(stress['data.shuffle']))
        self.assertEqual(json.loads(stress['trainer.test_freq']), 10)


if __name__ == '__main__':
    unittest.main()
