"""Recovery regressions use a virtual clock; no minutes-long test sleeps."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from search_r1_framework.config import Settings, tool_config
from search_r1_framework.search_wait import (
    wait_for_search_call, RetrieverInstanceTracker, SEARCH_RESULT_TIMEOUT_SECONDS,
    SEARCH_BRIDGE_TIMEOUT_SECONDS, SEARCH_HTTP_TIMEOUT_SECONDS, SEARCH_CANCEL_TIMEOUT_SECONDS)
from search_r1_framework.performance import PerformanceProbe, write_report
from search_r1_framework.smoke import validate_completed_smoke
from dataclasses import replace
from search_r1_framework.retrieval_protocol import retrieve_batch


class PollTimeout(Exception):
    pass


class FakeCall:
    def __init__(self, ready_at=200, error=None):
        self.now = 0
        self.ready_at = ready_at
        self.error = error
        self.object_id = "fc-same-call"
        self.polls, self.cancellations = [], 0
        self.get = SimpleNamespace(aio=self.read)
        self.cancel = SimpleNamespace(aio=self.stop)

    async def read(self, timeout):
        self.polls.append((self.object_id, timeout))
        if self.error:
            raise self.error
        if self.now + timeout >= self.ready_at:
            self.now = self.ready_at
            return {"result": [[]], "retriever_instance_id": "new"}
        self.now += timeout
        raise PollTimeout()

    async def stop(self):
        self.cancellations += 1


class SearchWaitTests(unittest.TestCase):
    def test_backend_timeout_cannot_be_mistaken_for_sdk_poll_timeout(self):
        def fail(*args):
            raise TimeoutError("index read failed")
        with self.assertRaisesRegex(RuntimeError, "Retriever execution.*index read failed"):
            retrieve_batch(SimpleNamespace(retrieve_many=fail), ["query"], 3)

    def test_reuse_only_successful_matching_smoke(self):
        settings = replace(Settings(), steps=2, eval_every=1, save_every=1)
        result = {"job": "train_grpo", "preparation_name": "prep", "settings": settings.to_dict(),
                  "smoke_dev_questions": 4, "adapter_path": "/results/run/checkpoints/global_step_2/actor/lora_adapter"}
        self.assertTrue(validate_completed_smoke(result, "train_grpo", "prep", settings))
        self.assertFalse(validate_completed_smoke({"status": "not_written_yet"}, "train_grpo", "prep", settings))
        for changed in ({"job": "train_ppo"}, {"settings": Settings().to_dict()}, {"adapter_path": None}):
            with self.assertRaises(ValueError):
                validate_completed_smoke({**result, **changed}, "train_grpo", "prep", settings)

    def run_call(self, call, events):
        return asyncio.run(wait_for_search_call(call, PollTimeout,
            lambda kind, detail: events.append((kind, detail)), clock=lambda: call.now))

    def test_recovery_waits_on_same_call_without_cancel_or_resubmit(self):
        call, events = FakeCall(ready_at=240), []
        self.assertEqual(self.run_call(call, events)["result"], [[]])
        self.assertGreater(len(call.polls), 1)
        self.assertEqual({item[0] for item in call.polls}, {"fc-same-call"})
        self.assertEqual(call.cancellations, 0)
        self.assertEqual([e[0] for e in events], ["extended_wait", "request_recovered"])

    def test_total_deadline_cancels_exactly_once_and_reports_call_id(self):
        call, events = FakeCall(ready_at=1000), []
        with self.assertRaisesRegex(TimeoutError, "fc-same-call.*600s"):
            self.run_call(call, events)
        self.assertEqual(call.now, 600)
        self.assertEqual(sum(x[1] for x in call.polls), 600)
        self.assertEqual(call.cancellations, 1)

    def test_real_backend_error_is_not_treated_as_poll_timeout(self):
        for error in (ValueError("invalid request"), RuntimeError("missing dependency"), TimeoutError("backend failed")):
            call = FakeCall(error=error)
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.run_call(call, [])
            self.assertEqual(len(call.polls), 1)
            self.assertEqual(call.cancellations, 1)

    def test_caller_cancellation_cancels_remote_call(self):
        call = FakeCall(error=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            self.run_call(call, [])
        self.assertEqual(call.cancellations, 1)

    def test_cancellation_failure_preserves_original_error(self):
        call = FakeCall(error=ValueError("original"))
        async def failed_cancel():
            raise RuntimeError("cannot cancel")
        call.cancel.aio = failed_cancel
        events = []
        with self.assertRaisesRegex(ValueError, "original"):
            self.run_call(call, events)
        self.assertEqual(events[0][0], "cancellation_failed")

    def test_normal_call_does_not_get_marked_as_recovery(self):
        call, events = FakeCall(ready_at=60), []
        self.run_call(call, events)
        self.assertFalse(events)
        self.assertEqual(call.cancellations, 0)

    def test_container_replacement_is_separate_from_slow_request(self):
        tracker = RetrieverInstanceTracker("original")
        events = []
        notify = lambda *event: events.append(event)
        tracker.observe({"retriever_instance_id": "original"}, notify)
        self.assertFalse(events)
        tracker.observe({"retriever_instance_id": "replacement"}, notify)
        tracker.observe({"retriever_instance_id": "replacement"}, notify)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "retriever_restarted")
        with self.assertRaisesRegex(RuntimeError, "redeploy"):
            tracker.observe({}, notify)

    def test_timeout_layers_leave_cleanup_margin(self):
        self.assertLess(SEARCH_RESULT_TIMEOUT_SECONDS + SEARCH_CANCEL_TIMEOUT_SECONDS, SEARCH_BRIDGE_TIMEOUT_SECONDS)
        self.assertLess(SEARCH_BRIDGE_TIMEOUT_SECONDS + SEARCH_CANCEL_TIMEOUT_SECONDS, SEARCH_HTTP_TIMEOUT_SECONDS)
        self.assertEqual(tool_config()["tools"][0]["config"]["timeout"], SEARCH_HTTP_TIMEOUT_SECONDS)

    def test_interrupted_performance_is_not_a_successful_comparison(self):
        rows = [{"step": step, "wall_time": step * 10, "metrics": {"timing_s/step": 10}}
                for step in (1, 2, 3)]
        probe = PerformanceProbe(warmup_steps=1, measured_steps=2)
        sampler = SimpleNamespace(samples=[], errors=[])
        for event in (None, "extended_wait", "retriever_restarted"):
            with self.subTest(event=event), tempfile.TemporaryDirectory() as directory:
                if event:
                    (Path(directory) / "search_requests.jsonl").write_text(json.dumps({"kind": event}) + "\n")
                with patch("search_r1_framework.performance.read_native_metrics", return_value=rows):
                    report = write_report(directory, probe, Settings(), "grpo", sampler, 30)
                self.assertEqual(report["status"], "completed_with_warnings" if event == "retriever_restarted" else "completed")
                self.assertEqual(report["comparison_eligible"], event != "retriever_restarted")
                self.assertEqual(report["measurement"]["mean_step_seconds"], 10)


if __name__ == "__main__":
    unittest.main()
