"""Local concurrent-call regressions; no Modal, models, or index required."""
import asyncio
import unittest

from search_r1_framework.shared_retrieval import SharedRetrievalQueue
from search_r1_framework.search_batcher import SearchBatcher


def response(queries):
    return {"result": [[{"document": {"contents": q}, "score": 1.0}] for q in queries],
            "retrieval_metrics": {"index_search_seconds": 1}}


class SharedRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_training_bridges_share_one_backend_batch(self):
        calls = []
        async def search(q, k):
            calls.append(q)
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=.01, log=lambda s: None)
        ppo = SearchBatcher(queue, wait_seconds=0)
        grpo = SearchBatcher(queue, wait_seconds=0)
        try:
            outputs = await asyncio.gather(ppo(["a"], 3), ppo(["b"], 3),
                                           grpo(["c"], 3), grpo(["d"], 3))
            self.assertEqual(calls, [["a", "b", "c", "d"]])
            self.assertEqual([r["result"][0][0]["document"]["contents"]
                              for r in outputs], ["a", "b", "c", "d"])
            self.assertEqual(sum(r["retrieval_metrics"].get("index_search_seconds", 0)
                                 for r in outputs), 1)
            self.assertTrue(all(r["shared_search_batch"]["unique_queries"] == 4
                                for r in outputs))
        finally:
            await ppo.close()
            await grpo.close()
            await queue.close()

    async def test_four_queries_trigger_early_and_split_in_order(self):
        calls = []
        async def search(q, k):
            calls.append((q, k))
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=10, log=lambda s: None)
        try:
            outputs = await asyncio.wait_for(asyncio.gather(
                queue(["a", "b", "a"], 3), queue(["c", "d"], 3)), 1)
            self.assertEqual(calls, [(["a", "b", "c", "d"], 3)])
            self.assertEqual([[g[0]["document"]["contents"] for g in r["result"]]
                             for r in outputs], [["a", "b", "a"], ["c", "d"]])
            self.assertEqual(sum(r["retrieval_metrics"].get("index_search_seconds", 0)
                                 for r in outputs), 1)
        finally:
            await queue.close()

    async def test_lone_query_deadline_not_extended_by_arrivals(self):
        entered = asyncio.Event()
        async def search(q, k):
            entered.set()
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=.12, target_queries=100,
                                     log=lambda s: None)
        tasks = [asyncio.create_task(queue(["first"], 3))]
        try:
            # Additional arrivals span the deadline and must not reset it.
            for i in range(3):
                await asyncio.sleep(.03)
                tasks.append(asyncio.create_task(queue([str(i)], 3)))
            await asyncio.wait_for(entered.wait(), .08)
            await asyncio.gather(*tasks)
        finally:
            await queue.close()

    async def test_cache_hits_bypass_busy_index_and_returns_are_isolated(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def search(q, k):
            calls.append(q)
            if q == ["slow"]:
                entered.set()
                await release.wait()
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=0, log=lambda s: None)
        try:
            first = await queue(["cached"], 3)
            first["result"][0][0]["document"]["contents"] = "modified"
            slow = asyncio.create_task(queue(["slow"], 3))
            await entered.wait()
            hit = await asyncio.wait_for(queue(["cached"], 3), .1)
            self.assertEqual(hit["result"][0][0]["document"]["contents"], "cached")
            self.assertEqual(hit["shared_search_batch"]["front_cache_hit_unique_queries"], 1)
            self.assertEqual(calls, [["cached"], ["slow"]])
            release.set()
            await slow
        finally:
            release.set()
            await queue.close()

    async def test_backlog_runs_without_fresh_window_and_no_overlapping_search(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def search(q, k):
            calls.append(q)
            if len(calls) == 1:
                entered.set()
                await release.wait()
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=10, log=lambda s: None)
        try:
            first = asyncio.create_task(queue(["a", "b", "c", "d"], 3))
            await entered.wait()
            second = asyncio.create_task(queue(["queued"], 3))
            await asyncio.sleep(0)
            self.assertEqual(len(calls), 1)
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 1)
            self.assertEqual(calls[1], ["queued"])
        finally:
            release.set()
            await queue.close()

    async def test_topk_cache_keys_and_partial_hits(self):
        calls = []
        async def search(q, k):
            calls.append((q, k))
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=0, log=lambda s: None)
        try:
            await queue(["a"], 3)
            await queue(["a", "b", "a"], 3)
            await queue(["a"], 1)
            self.assertEqual(calls, [(["a"], 3), (["b"], 3), (["a"], 1)])
        finally:
            await queue.close()

    async def test_failed_backend_reaches_both_callers_and_does_not_cache(self):
        fail = True
        async def search(q, k):
            if fail:
                raise RuntimeError("index failure")
            return response(q)
        queue = SharedRetrievalQueue(search, wait_seconds=.001, log=lambda s: None)
        try:
            results = await asyncio.gather(queue(["a"], 3), queue(["b"], 3),
                                           return_exceptions=True)
            self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
            self.assertFalse(queue.cache)
            fail = False
            self.assertEqual(len((await queue(["a"], 3))["result"]), 1)
        finally:
            await queue.close()

    async def test_cancel_one_caller_preserves_other_callers(self):
        queue = SharedRetrievalQueue(lambda q, k: asyncio.sleep(0, result=response(q)),
                                     wait_seconds=.01, log=lambda s: None)
        try:
            cancelled = asyncio.create_task(queue(["cancel"], 3))
            keep = asyncio.create_task(queue(["keep"], 3))
            await asyncio.sleep(0)
            cancelled.cancel()
            result = await asyncio.wait_for(keep, 1)
            self.assertEqual(result["result"][0][0]["document"]["contents"], "keep")
            await asyncio.gather(cancelled, return_exceptions=True)
        finally:
            await queue.close()


if __name__ == "__main__":
    unittest.main()
