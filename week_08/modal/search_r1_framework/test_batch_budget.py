"""CPU regressions for real FAISS batching and long-running performance jobs.

Uses a deterministic test encoder, not a downloaded language model or corpus.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from search_r1_framework.config import Settings
from search_r1_framework.performance import (PerformanceProbe, write_report,
    PERFORMANCE_HARD_TIMEOUT_SECONDS)
from search_r1_framework.runner import run_framework


class BatchTests(unittest.TestCase):
    def test_real_faiss_matches_single_queries_across_chunk_boundaries(self):
        import numpy as np
        import torch
        import faiss
        from search_r1_retriever import DenseRetriever

        class Tokenizer:
            def __call__(self, strings, **kwargs):
                values = [[ord(c) for c in s][:256] for s in strings]
                length = max(map(len, values))
                return {"input_ids": torch.tensor([v + [0] * (length - len(v)) for v in values]),
                        "attention_mask": torch.tensor([[1] * len(v) + [0] * (length - len(v)) for v in values])}

        class Encoder:
            def __call__(self, input_ids, **kwargs):
                # Padded tokens have nonzero embeddings: masking must remove them.
                hidden = torch.cos(input_ids.float().unsqueeze(-1) * torch.arange(1, 9) / 17)
                return SimpleNamespace(last_hidden_state=hidden)

        index = faiss.IndexFlatIP(8)
        vectors = np.random.default_rng(42).normal(size=(101, 8)).astype('float32')
        faiss.normalize_L2(vectors)
        index.add(vectors)
        retriever = DenseRetriever.__new__(DenseRetriever)
        retriever.index = index
        retriever.tokenizer, retriever.encoder = Tokenizer(), Encoder()
        retriever.corpus = [{"contents": f"passage-{i}"} for i in range(101)]
        queries = [f"query {i}" + " long" * (i % 6) for i in range(35)]
        queries += [queries[3], queries[0]]
        expected = [retriever.retrieve(q, 3) for q in queries]
        calls = []
        def search(vectors, topk):
            calls.append(len(vectors))
            return index.search(vectors, topk)
        retriever.index = SimpleNamespace(d=8, search=search)
        actual = retriever.retrieve_many(queries, 3, use_cache=False)
        self.assertEqual(calls, [32, 3])
        self.assertEqual([[x['id'] for x in g] for g in actual],
                         [[x['id'] for x in g] for g in expected])
        np.testing.assert_allclose([[x['score'] for x in g] for g in actual],
                                   [[x['score'] for x in g] for g in expected], atol=2e-6)
        self.assertEqual(retriever.last_batch_metrics['unique_query_count'], 35)
        self.assertEqual(retriever.last_batch_metrics['query_count'], 37)
        self.assertGreaterEqual(retriever.last_batch_metrics['index_search_seconds'], 0)
        retriever.clear_query_cache()
        calls.clear()
        cold = retriever.retrieve_many(["repeat", "repeat", "second"], 3)
        self.assertEqual(calls, [2])
        warm = retriever.retrieve_many(["second", "repeat", "repeat"], 3)
        self.assertEqual(calls, [2])
        self.assertEqual(warm, [cold[2], cold[0], cold[0]])
        self.assertEqual(retriever.last_batch_metrics['cache_hit_queries'], 3)
        self.assertEqual(retriever.last_batch_metrics['searched_unique_queries'], 0)
        self.assertEqual(retriever.last_batch_metrics['index_search_seconds'], 0)
        warm[0][0]['contents'] = 'caller mutation'
        self.assertNotEqual(retriever.retrieve('second', 3)[0]['contents'], 'caller mutation')
        retriever.retrieve('second', 1)
        self.assertEqual(calls, [2, 1])  # top-k is part of the cache key.
        retriever.retrieve('Second', 3)
        self.assertEqual(calls, [2, 1, 1])  # Query text is never normalized.
        retriever.clear_query_cache()
        retriever.QUERY_CACHE_MAX_ENTRIES = 2
        retriever.retrieve_many(['a', 'b'], 3)
        retriever.retrieve('a', 3)
        retriever.retrieve('c', 3)
        self.assertEqual(list(retriever._query_cache), [('a', 3), ('c', 3)])
        before = len(calls)
        retriever.retrieve('b', 3)
        self.assertEqual(len(calls), before + 1)
        self.assertEqual(len(retriever._query_cache), 2)
        with patch.object(retriever.index, 'search', side_effect=RuntimeError('index failed')):
            with self.assertRaisesRegex(RuntimeError, 'index failed'):
                retriever.retrieve('failed query', 3)
        self.assertNotIn(('failed query', 3), retriever._query_cache)
        index.reset()
        self.assertEqual(retriever.retrieve_many(['no documents'], 3), [[]])


class ProgressTests(unittest.TestCase):
    def test_worker_continues_past_one_hour_and_keeps_progress(self):
        self.assertEqual(PERFORMANCE_HARD_TIMEOUT_SECONDS, 86400)
        process = MagicMock(stdout=iter(["step 6 completed\n", "step 10 completed\n"]))
        process.wait.return_value = 0
        process.poll.return_value = 0
        snapshots = []
        ticks = iter(range(0, 100000, 4000))
        with tempfile.TemporaryDirectory() as directory, \
             patch('search_r1_framework.runner.subprocess.run', return_value=SimpleNamespace(stdout='config')), \
             patch('search_r1_framework.runner.subprocess.Popen', return_value=process), \
             patch('search_r1_framework.runner.time.monotonic', side_effect=lambda: next(ticks)), \
             patch('builtins.print'):
            run_framework([], directory, progress=lambda: snapshots.append(True))
            self.assertTrue(snapshots)
            self.assertIn('step 10 completed', (Path(directory) / 'console.log').read_text())
            process.terminate.assert_not_called()
            process.kill.assert_not_called()

    def test_partial_snapshot_is_kept_without_claiming_completion(self):
        rows = [{"step": 3, "wall_time": 100, "metrics": {
            "timing_s/step": 30, "timing_s/gen": 25, "timing_s/update_actor": 3}}]
        sampler = SimpleNamespace(samples=[], errors=[])
        with tempfile.TemporaryDirectory() as directory, \
             patch('search_r1_framework.performance.read_native_metrics', return_value=rows):
            report = write_report(directory, PerformanceProbe(), Settings(), 'grpo', sampler, 7200, running=True)
            self.assertEqual(report['status'], 'running')
            self.assertIsNone(report['work_budget_seconds'])
            self.assertEqual(report['completed_steps'], [3])
            self.assertFalse(report['comparison_eligible'])
            self.assertEqual(json.loads((Path(directory) / 'performance.json').read_text())['status'], 'running')


if __name__ == '__main__':
    unittest.main()
