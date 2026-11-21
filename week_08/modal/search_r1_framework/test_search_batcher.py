"""CPU-only contract and completion regressions; no cloud work is submitted."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import contextlib
from dataclasses import replace
import sys
from unittest.mock import patch

from search_r1_framework.search_batcher import SearchBatcher, RETRIEVAL_EXECUTION
from search_r1_framework.performance import (PerformanceProbe, write_report,
    execution_completed, validate_reusable_report)
from search_r1_framework.config import Settings


class SearchBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_merge_order_duplicates_and_metrics_counted_once(self):
        calls = []
        async def backend(queries, topk):
            calls.append((queries, topk))
            return {"result": [[{"document": {"contents": q}, "score": 1.0}] for q in queries],
                    "retrieval_metrics": {"index_search_seconds": 3}, "retriever_instance_id": "one"}
        batch = SearchBatcher(backend, wait_seconds=0.001)
        results = await asyncio.gather(batch(["A", "B"], 3), batch(["A"], 3), batch(["C"], 3))
        self.assertEqual(calls, [(["A", "B", "A", "C"], 3)])
        self.assertEqual([[g[0]["document"]["contents"] for g in r["result"]] for r in results],
                         [["A", "B"], ["A"], ["C"]])
        self.assertEqual(sum(r["retrieval_metrics"].get("index_search_seconds", 0) for r in results), 3)
        self.assertTrue(all(r["retriever_instance_id"] == "one" for r in results))
        await batch.close()

    async def test_cap_topk_separation_and_cancelled_request(self):
        calls = []
        async def backend(queries, topk):
            calls.append((queries, topk))
            return {"result": [[] for q in queries]}
        batch = SearchBatcher(backend, wait_seconds=0.001, max_queries=2)
        tasks = [asyncio.create_task(batch(q, k)) for q, k in
                 [(["a", "b"], 3), (["c"], 3), (["d"], 1), (["cancel"], 1)]]
        await asyncio.sleep(0)
        tasks[-1].cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(calls, [(["a", "b"], 3), (["c"], 3), (["d"], 1)])
        await batch.close()

    async def test_backend_failure_reaches_all_waiters(self):
        async def backend(queries, topk):
            raise RuntimeError("backend died")
        batch = SearchBatcher(backend, wait_seconds=0.001, max_queries=1)
        results = await asyncio.gather(batch(["a"], 3), batch(["b"], 3), return_exceptions=True)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
        await batch.close()

    async def test_shutdown_cancels_waiting_requests(self):
        batch = SearchBatcher(None, wait_seconds=10)
        task = asyncio.create_task(batch(["a"], 3))
        await asyncio.sleep(0)
        await batch.close()
        with self.assertRaises(asyncio.CancelledError):
            await task


class CompletionTests(unittest.TestCase):
    def make_report(self, error=None, restart=True):
        base = Settings()
        probe = PerformanceProbe(warmup_steps=2, measured_steps=2)
        settings = probe.settings_for(base)
        rows = [{"step": s, "wall_time": s * 10,
                 "metrics": {"timing_s/step": 10}} for s in range(1, 5)]
        with tempfile.TemporaryDirectory() as d:
            if restart:
                Path(d, 'search_requests.jsonl').write_text(json.dumps({"kind": "retriever_restarted"})+'\n')
            with patch('search_r1_framework.performance.read_native_metrics', return_value=rows):
                report = write_report(d, probe, settings, "grpo", SimpleNamespace(samples=[], errors=[]), 50, error)
        report['preparation_name'] = 'prep'
        return report, base, probe

    def test_recovered_completion_is_warning_not_failure(self):
        report, base, probe = self.make_report()
        self.assertEqual(report['status'], 'completed_with_warnings')
        self.assertTrue(execution_completed(report))
        self.assertFalse(report['comparison_eligible'])
        validate_reusable_report(report, 'grpo', 'prep', base, probe)
        report['status'] = 'interrupted_measurement'
        self.assertTrue(execution_completed(report))
        report['completed_steps'].pop()
        self.assertFalse(execution_completed(report))

    def test_real_failure_and_changed_settings_rejected(self):
        report, base, probe = self.make_report(RuntimeError('worker failed'))
        self.assertFalse(execution_completed(report))
        with self.assertRaises(ValueError):
            validate_reusable_report(report, 'grpo', 'prep', base, probe)
        report, base, probe = self.make_report(restart=False)
        self.assertTrue(report['comparison_eligible'])
        for key, value in [('retrieval_execution', 'old'), ('preparation_name', 'other'),
                           ('algorithm', 'ppo')]:
            changed = dict(report, **{key: value})
            with self.assertRaises(ValueError):
                validate_reusable_report(changed, 'grpo', 'prep', base, probe)


class NotebookTests(unittest.TestCase):
    def run_plan(self, notebook, fail_second=False):
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        plan = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code'
                    and 'performance_plan = []' in ''.join(c['source']))
        comparison = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code'
                          and 'COMPARE_PERFORMANCE_TESTS = True' in ''.join(c['source']))
        saved, calls = {}, []
        def spawn(algorithm, name, preparation, settings, probe):
            calls.append(name)
            if fail_second and len(calls) == 2:
                raise RuntimeError('simulated cloud failure')
            effective = PerformanceProbe(**probe).settings_for(Settings(**settings))
            saved[name] = {'algorithm': algorithm, 'preparation_name': preparation,
                          'probe': probe, 'settings': effective.to_dict(),
                          'retrieval_execution': RETRIEVAL_EXECUTION,
                          'status': 'completed_with_warnings', 'error': None,
                          'completed_steps': list(range(1, effective.steps+1)),
                          'measurement': {'complete_measurement': True, 'mean_step_seconds': 1},
                          'comparison_eligible': False}
            return SimpleNamespace(object_id=name)
        fake_modal = SimpleNamespace(enable_output=contextlib.nullcontext,
                    Function=SimpleNamespace(from_name=lambda *a: SimpleNamespace(spawn=spawn)))
        env = dict(Settings=Settings, PerformanceProbe=PerformanceProbe, settings=Settings(),
                   replace=replace, APP_NAME='app', PREPARATION='prep', json=json, display=lambda x: None,
                   read_output=lambda name, *args: saved.get(name, {'status': 'not_written_yet'}),
                   get_finished_call=lambda name: saved[name])
        with patch.dict(sys.modules, {'modal': fake_modal}), patch('builtins.print'):
            if fail_second:
                with self.assertRaisesRegex(RuntimeError, 'simulated cloud failure'):
                    exec(plan, env)
                self.assertEqual(len(calls), 2)
            else:
                exec(plan, env)
                self.assertEqual(len(calls), 6)
                self.assertTrue(all(r['settings']['steps'] == 4 for r in saved.values()))
                exec(comparison, env)  # No clean runs: display warnings, do not crash.
                exec(plan, env)  # Completed warning reports are reused without submission.
                self.assertEqual(len(calls), 6)

    def test_both_notebooks_continue_on_warning_reuse_and_stop_on_failure(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('assignment_08_tests.ipynb', 'assignment_08_tests_zh.ipynb'):
            with self.subTest(notebook=name):
                self.run_plan(root / name)
                self.run_plan(root / name, fail_second=True)


if __name__ == '__main__':
    unittest.main()
