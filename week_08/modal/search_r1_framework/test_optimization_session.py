"""Local simulations of paid-session orchestration; no Modal calls are made."""
import contextlib
import asyncio
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, AsyncMock, patch

import run_optimization_tests as session
from search_r1_framework import runner
from search_r1_framework.performance import PerformanceProbe
from search_r1_framework.config import Settings
from search_r1_framework.search_batcher import RETRIEVAL_EXECUTION
from search_r1_framework.search_wait import poll_function_call


class RemoteExecutionTimeout(Exception):
    """Distinct from the builtin pending-result timeout, as in the actual SDK."""


def report(algo, micro):
    probe = PerformanceProbe(micro_batch=micro, warmup_steps=2, measured_steps=2)
    return {"status": "completed", "error": None, "completed_steps": [1, 2, 3, 4],
            "settings": probe.settings_for(Settings()).to_dict(), "probe": probe.to_dict(),
            "algorithm": algo, "preparation_name": session.PREPARATION,
            "retrieval_execution": RETRIEVAL_EXECUTION,
            "measurement": {"complete_measurement": True}}


class OptimizationSessionTests(unittest.TestCase):
    def exercise(self, oom=False, pending_polls=0):
        files = {f"perf-{algo}-mb8-v4/performance.json": json.dumps(report(algo, 8)).encode()
                 for algo in ("grpo", "ppo")}
        for run, kind in (("perf-grpo-mb2-v4", "retriever_restarted"),
                          ("perf-grpo-mb2-v3", "extended_wait")):
            files[run + "/search_requests.jsonl"] = json.dumps(
                {"kind": kind, "request": {"queries": [f"query{i}" for i in range(32)]}}).encode()
        submitted = []
        calls = []

        def call(result, fail=False):
            c = Mock(object_id=f"fc-{len(calls)}")
            c.get.return_value = result
            if fail:
                c.get.side_effect = RuntimeError("CUDA out of memory")
            calls.append(c)
            return c

        def train(algo, run, prep, settings, probe, **kwargs):
            submitted.append((algo, probe["micro_batch"]))
            self.assertNotIn("experiment_deadline_unix", kwargs)
            r = report(algo, probe["micro_batch"])
            fail = oom and algo == "grpo" and probe["micro_batch"] == 16
            if fail:
                r.update(status="failed", error="CUDA out of memory", completed_steps=[])
                r["measurement"]["complete_measurement"] = False
                files[run + "/console.log"] = b"torch.OutOfMemoryError: CUDA out of memory"
            files[run + "/performance.json"] = json.dumps(r).encode()
            return call(r, fail)

        def retrieve(queries, topk):
            result = {"retriever_instance_id": "one",
                         "result": [[{"document": {"id": q + str(k)}, "score": 1.0}
                                     for k in range(topk)] for q in queries]}
            c = call(result)
            if len(calls) == 1 and pending_polls:
                c.get.side_effect = [TimeoutError() for _ in range(pending_polls)] + [result]
            return c

        volume = SimpleNamespace(read_file=lambda p: iter([files[p]]), iterdir=lambda *a, **k: [])
        fake_modal = SimpleNamespace(
            Volume=SimpleNamespace(from_name=lambda n: volume),
            Function=SimpleNamespace(from_name=lambda *a: SimpleNamespace(spawn=train)),
            Cls=SimpleNamespace(from_name=lambda *a: lambda: SimpleNamespace(retrieve=SimpleNamespace(spawn=retrieve))),
            exception=SimpleNamespace(TimeoutError=RemoteExecutionTimeout))
        with tempfile.TemporaryDirectory() as tmp, patch.object(session, "OUT", Path(tmp)/"results"), \
             patch.dict("sys.modules", {"modal": fake_modal}), \
             patch.object(session.subprocess, "run", return_value=SimpleNamespace(stdout="deployed")) as cli, \
             contextlib.redirect_stdout(io.StringIO()):
            session.main(["--execute"])
            saved = json.loads((session.OUT/"summary.json").read_text())
            self.assertTrue(saved["app_stopped"])
            self.assertEqual(len(saved["retrieval"]), 6)
            self.assertTrue(all(x["same_ranked_document_ids"] for x in saved["retrieval"]))
            self.assertEqual(cli.call_args_list[-1].args[0][-4:], ["app", "stop", session.APP, "--yes"])
            self.assertEqual(calls[0].get.call_count, pending_polls + 1)
            calls[0].cancel.assert_not_called()
            # Run All a second time must not redeploy or start paid work.
            before = len(submitted)
            with self.assertRaises(FileExistsError):
                session.main(["--execute"])
            self.assertEqual(len(submitted), before)
        return submitted

    def test_full_plan_and_no_duplicate_execution(self):
        self.assertEqual(self.exercise(), [("grpo",16),("ppo",16),("grpo",32),("ppo",32)])

    def test_cold_start_polls_same_call_without_cancelling_or_resubmitting(self):
        self.assertEqual(self.exercise(pending_polls=3), [("grpo",16),("ppo",16),("grpo",32),("ppo",32)])

    def test_remote_execution_timeout_is_not_treated_as_pending(self):
        call, notify = Mock(), Mock()
        call.get.side_effect = RemoteExecutionTimeout("remote execution timed out")
        with self.assertRaises(RemoteExecutionTimeout):
            poll_function_call(call, notify)
        self.assertEqual(call.get.call_count, 1)
        notify.assert_not_called()

    def test_pending_polls_continue_beyond_an_hour_without_cancellation(self):
        call, notify = Mock(), Mock()
        call.get.side_effect = [TimeoutError() for _ in range(121)] + ["ready"]
        self.assertEqual(poll_function_call(call, notify), "ready")
        self.assertEqual(call.get.call_count, 122)
        self.assertEqual(notify.call_count, 121)
        self.assertTrue(all(x.kwargs["timeout"] == 30 for x in call.get.call_args_list))
        call.cancel.assert_not_called()

    def test_installed_modal_sdk_pending_exception(self):
        try:
            import modal
            from modal._functions import _Invocation
        except ImportError:
            self.skipTest("Requires the user's installed Modal SDK; no network calls.")
        self.assertIsNot(modal.exception.TimeoutError, TimeoutError)
        invocation = SimpleNamespace(pop_function_call_outputs=AsyncMock(
            return_value=SimpleNamespace(outputs=[], num_unfinished_inputs=1)))
        def actual_pending_poll(**kwargs):
            return asyncio.run(_Invocation.poll_function(invocation, **kwargs))
        call = Mock()
        count = [0]
        def get(**kwargs):
            count[0] += 1
            if count[0] < 3:
                return actual_pending_poll(**kwargs)
            return "ready"
        call.get.side_effect = get
        notify = Mock()
        self.assertEqual(poll_function_call(call, notify), "ready")
        self.assertEqual(notify.call_count, 2)
        call.cancel.assert_not_called()
        call.get.side_effect = modal.exception.TimeoutError("remote timeout")
        with self.assertRaises(modal.exception.TimeoutError):
            poll_function_call(call, notify)

    def test_oom_skips_only_larger_batch_for_same_algorithm(self):
        self.assertEqual(self.exercise(oom=True), [("grpo",16),("ppo",16),("ppo",32)])

    def test_cost_estimate_never_blocks_the_plan(self):
        with patch.object(session, "HOURLY_ESTIMATE", 1000):
            self.assertEqual(self.exercise(), [("grpo",16),("ppo",16),("grpo",32),("ppo",32)])

    def test_cloud_runner_has_no_experiment_timeout(self):
        process = Mock(pid=123, stdout=iter([]))
        process.poll.return_value = 0
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="config")) as compose, \
             patch.object(runner.subprocess, "Popen", return_value=process):
            runner.run_framework([], tmp)
            self.assertNotIn("timeout", compose.call_args.kwargs)
            process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
