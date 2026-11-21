"""Formal notebook configuration and sequencing checks with simulated Modal calls."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from search_r1_framework.config import Settings, overrides

ROOT = Path(__file__).resolve().parents[1]


class FormalWorkflowTests(unittest.TestCase):
    def test_progress_handles_startup_and_unreadable_metrics(self):
        from search_r1_framework.performance import atomic_json
        tree = ast.parse((ROOT / 'search_r1_framework_modal.py').read_text(encoding="utf-8"))
        job = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_job')
        callback = next(n for n in job.body if isinstance(n, ast.FunctionDef) and n.name == 'formal_progress')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reader, volume = MagicMock(), MagicMock()
            scope = {'output': output, 'job': 'train_grpo', 'run_name': 'formal-test',
                     's': Settings(), 'time': time, 'job_started': time.monotonic(),
                     'read_native_metrics': reader, 'atomic_json': atomic_json, 'results': volume,
                     'algorithm': 'grpo', 'seal_latest_checkpoint': lambda *a: None}
            exec(compile(ast.Module(body=[callback], type_ignores=[]), '<progress>', 'exec'), scope)
            with patch('builtins.print'):
                scope['formal_progress']()
                reader.assert_not_called()
                (output / 'tensorboard').mkdir()
                reader.side_effect = OSError('incomplete event file')
                scope['formal_progress']()
            progress = json.loads((output / 'progress.json').read_text(encoding="utf-8"))
            self.assertEqual(progress['status'], 'running')
            self.assertEqual(progress['completed_training_steps'], 0)
            self.assertIn('incomplete event', progress['metrics_read_error'])
            self.assertEqual(volume.commit.call_count, 2)

    def test_selected_settings_reach_both_algorithms_and_evaluation(self):
        settings = Settings()
        for algorithm in ('ppo', 'grpo'):
            for evaluation in (False, True):
                args = dict(a.split('=', 1) for a in overrides(
                    settings, algorithm, '/data/p', '/results/r', '/model', '/actor', '/critic', evaluation=evaluation))
                self.assertEqual(json.loads(args['data.max_prompt_length']), 384)
                self.assertEqual(json.loads(args['data.max_response_length']), 2560)
                self.assertEqual(json.loads(args['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']), 8)
                self.assertEqual(json.loads(args['critic.ppo_micro_batch_size_per_gpu']), 8)
                self.assertEqual(json.loads(args['trainer.total_training_steps']), 100)
        tree = ast.parse((ROOT / 'search_r1_framework_modal.py').read_text(encoding="utf-8"))
        constant = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FORMAL_GPU' for t in n.targets))
        self.assertEqual(ast.literal_eval(constant.value), 'H200')
        job = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_job')
        gpu = next(k.value for k in job.decorator_list[0].keywords if k.arg == 'gpu')
        self.assertEqual(gpu.id, 'FORMAL_GPU')
        containers = next(k.value for k in job.decorator_list[0].keywords if k.arg == 'max_containers')
        self.assertEqual(ast.literal_eval(containers), 2)

    def test_notebook_persists_call_and_waits_without_resubmission(self):
        notebook = json.loads((ROOT / 'assignment_08_verl.ipynb').read_text(encoding="utf-8"))
        source = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('def submit('))
        remote = MagicMock()
        remote.get.side_effect = [TimeoutError(), {'completed': True}]
        mock_modal = SimpleNamespace(Function=MagicMock(), FunctionCall=MagicMock(), enable_output=nullcontext)
        mock_modal.Function.from_name.return_value.spawn.return_value.object_id = 'fc-test'
        mock_modal.FunctionCall.from_id.return_value = remote
        with tempfile.TemporaryDirectory() as directory:
            scope = {'settings': Settings(), 'APP_NAME': 'test', 'PREPARATION': 'prepared',
                     'Path': lambda name: Path(directory) / name, 'json': json}
            exec(compile(source, '<formal-helpers>', 'exec'), scope)
            with patch.dict('sys.modules', {'modal': mock_modal}), patch('builtins.print'):
                call_id = scope['submit']('train_grpo', 'formal-test')
                result = scope['get_finished_call'](call_id)
            self.assertEqual(result, {'completed': True})
            mock_modal.Function.from_name.return_value.spawn.assert_called_once()
            saved = json.loads((Path(directory) / 'formal_calls.jsonl').read_text(encoding="utf-8"))
            self.assertEqual(saved['call_id'], 'fc-test')
            self.assertEqual(saved['settings']['response_tokens'], 2560)
        for cell in notebook['cells']:
            code = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and '= submit(' in code and 'training_calls = {}' not in code:
                self.assertIn('= get_finished_call(', code)

    def test_both_training_jobs_submit_before_waiting(self):
        notebook = json.loads((ROOT / 'assignment_08_verl.ipynb').read_text(encoding="utf-8"))
        code = next(''.join(c['source']) for c in notebook['cells'] if 'training_calls = {}' in ''.join(c['source']))
        code = code.replace('RUN_PPO = False', 'RUN_PPO = True').replace('RUN_GRPO = False', 'RUN_GRPO = True')
        from dataclasses import replace
        events = []
        def submit(job, name, chosen_settings, resume):
            events.append((job, name, chosen_settings.steps))
            return f'fc-{job}'
        scope = {'settings': Settings(), 'PPO_RUN': 'ppo', 'GRPO_RUN': 'grpo', 'replace': replace,
                 'submit': submit, 'APP_NAME': 'test', 'PREPARATION': 'prepared', 'display': lambda x: None}
        mock_modal = SimpleNamespace(Function=MagicMock())
        with patch('builtins.print'), patch.dict('sys.modules', {'modal': mock_modal}):
            exec(compile(code, '<parallel-submit>', 'exec'), scope)
        self.assertEqual(events, [('train_ppo', 'ppo', 100), ('train_grpo', 'grpo', 100)])
        self.assertEqual(set(scope['training_calls']), {'ppo', 'grpo'})

    def test_parallel_wait_observes_both_and_does_not_cancel_on_failure(self):
        from search_r1_framework.parallel_jobs import wait_for_calls
        ppo, grpo = MagicMock(object_id='fc-ppo'), MagicMock(object_id='fc-grpo')
        ppo.get.side_effect = [TimeoutError(), {'ok': 'ppo'}]
        grpo.get.return_value = {'ok': 'grpo'}
        with patch('search_r1_framework.parallel_jobs.time.sleep'):
            result = wait_for_calls({'ppo': ppo, 'grpo': grpo}, notify=lambda _: None)
        self.assertEqual(result, {'grpo': {'ok': 'grpo'}, 'ppo': {'ok': 'ppo'}})
        ppo.get.side_effect = RuntimeError('failed')
        grpo.get.side_effect = TimeoutError()
        with self.assertRaisesRegex(RuntimeError, 'fc-grpo'):
            wait_for_calls({'ppo': ppo, 'grpo': grpo}, notify=lambda _: None)
        ppo.cancel.assert_not_called()
        grpo.cancel.assert_not_called()

    def test_notebook_can_resume_both_or_skip_finished_method(self):
        from dataclasses import replace
        notebook = json.loads((ROOT / 'assignment_08_verl.ipynb').read_text(encoding="utf-8"))
        original = next(''.join(c['source']) for c in notebook['cells']
                        if 'training_calls = {}' in ''.join(c['source']))
        for actions in ({'ppo': 'resume', 'grpo': 'resume'},
                        {'ppo': 'skip', 'grpo': 'resume'}):
            with self.subTest(actions=actions):
                code = original.replace('{"ppo": "new", "grpo": "new"}', repr(actions))
                submit = MagicMock(return_value='fc-new')
                remote = MagicMock()
                mock_modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *a: remote))
                scope = {'settings': Settings(), 'PPO_RUN': 'ppo', 'GRPO_RUN': 'grpo',
                         'replace': replace, 'submit': submit, 'APP_NAME': 'test',
                         'PREPARATION': 'prepared', 'display': lambda x: None}
                with patch.dict('sys.modules', {'modal': mock_modal}), patch('builtins.print'):
                    exec(code, scope)
                selected = [k for k, v in actions.items() if v == 'resume']
                self.assertEqual(submit.call_count, len(selected))
                self.assertTrue(all(c.kwargs['resume'] for c in submit.call_args_list))
                self.assertEqual(set(scope['training_calls']), set(selected))

    def test_failed_preflight_prevents_both_gpu_submissions(self):
        from dataclasses import replace
        notebook = json.loads((ROOT / 'assignment_08_verl.ipynb').read_text(encoding="utf-8"))
        code = next(''.join(c['source']) for c in notebook['cells'] if 'training_calls = {}' in ''.join(c['source']))
        submit, remote = MagicMock(), MagicMock()
        remote.remote.side_effect = [{'passed': True}, ValueError('missing checkpoint')]
        mock_modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *a: remote))
        scope = {'settings': Settings(), 'PPO_RUN': 'ppo', 'GRPO_RUN': 'grpo', 'replace': replace,
                 'submit': submit, 'APP_NAME': 'test', 'PREPARATION': 'prepared', 'display': lambda x: None}
        with patch.dict('sys.modules', {'modal': mock_modal}), self.assertRaises(ValueError):
            exec(code, scope)
        submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
