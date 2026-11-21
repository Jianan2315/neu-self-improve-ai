"""Protocol, HTTP failure handling, and runner cancellation regressions on CPU."""
import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from search_r1_framework.retrieval_protocol import validate_request, validate_response, retrieve_batch
from search_r1_framework.runner import run_framework


class ContractTests(unittest.TestCase):
    def test_valid_multi_query_order_and_empty_results(self):
        engine = SimpleNamespace(retrieve_many=lambda queries, k: [([] if q == "empty" else
            [{"id": q, "contents": q + "\nevidence", "score": 0.5}]) for q in queries])
        response = retrieve_batch(engine, ["first", "empty", "last"], 3)
        self.assertEqual([len(group) for group in response["result"]], [1, 0, 1])
        self.assertEqual(response["result"][2][0]["document"]["id"], "last")

    def test_invalid_inputs_are_not_coerced(self):
        for queries in ("text", [], None, [""], ["  "], [1], [["nested"]], [True]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                validate_request(queries, 3)
        for topk in ("3", True, 0, -1, 2.5):
            with self.subTest(topk=topk), self.assertRaises(ValueError):
                validate_request(["valid"], topk)

    def test_bad_response_is_rejected_before_native_formatter(self):
        for response in ({}, {"result": []}, {"result": ["text"]},
                         {"result": [[{"document": {"contents": None}, "score": 1}]]},
                         {"result": [[{"document": {"contents": "ok"}, "score": float("nan")}]]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_response(response, 1, 3)

    def test_backend_failure_interrupts_a_silent_child(self):
        failures = []
        original_popen = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            child = original_popen([sys.executable, "-u", "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            return child
        timer = threading.Timer(0.3, lambda: failures.append("retriever unavailable"))
        with tempfile.TemporaryDirectory() as directory, \
             patch("search_r1_framework.runner.subprocess.run", return_value=SimpleNamespace(stdout="config")), \
             patch("search_r1_framework.runner.subprocess.Popen", side_effect=launch):
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "retriever unavailable"):
                    run_framework([], directory, backend_failures=failures)
            finally:
                timer.cancel()
            self.assertLess(time.monotonic() - started, 8)
            self.assertIsNotNone(children[0].poll())


class HttpContractTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from search_r1_framework.search_bridge import create_search_api
        self.failures, self.calls = [], []
        self.response = {"result": [[]]}
        self.backend_error = None
        async def search(queries, topk):
            self.calls.append((queries, topk))
            if self.backend_error:
                raise self.backend_error
            return self.response
        self.client = TestClient(create_search_api(search, self.failures))

    def test_multi_query_and_invalid_request_classification(self):
        self.response = {"result": [[], []]}
        self.assertEqual(self.client.post("/retrieve", json={"queries": ["a", "b"], "topk": 3}).status_code, 200)
        for payload in ({}, {"queries": "a"}, {"queries": [None]}, {"queries": ["a"], "topk": "3"}):
            self.assertEqual(self.client.post("/retrieve", json=payload).status_code, 400)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.failures)

    def test_backend_failure_is_fatal_and_subsequent_requests_do_not_retry_rpc(self):
        self.backend_error = RuntimeError("engine crashed")
        for _ in range(2):
            self.assertEqual(self.client.post("/retrieve", json={"queries": ["a"]}).status_code, 503)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("engine crashed", self.failures[0])

    def test_malformed_backend_response_is_fatal(self):
        self.response = {"result": []}
        self.assertEqual(self.client.post("/retrieve", json={"queries": ["a"]}).status_code, 503)
        self.assertTrue(self.failures)

    def test_timeout_cancels_await_and_reports_fatal_failure(self):
        from fastapi.testclient import TestClient
        from search_r1_framework.search_bridge import create_search_api
        cancelled = []
        async def blocked(*args):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)
        client = TestClient(create_search_api(blocked, self.failures, timeout=0.03))
        self.assertEqual(client.post("/retrieve", json={"queries": ["a"]}).status_code, 503)
        self.assertEqual(cancelled, [True])
        self.assertIn("TimeoutError", self.failures[0])


if __name__ == "__main__":
    unittest.main()
