"""Coalesce concurrent tool requests without changing query order or top-k."""
import asyncio
from collections import deque

from search_r1_framework.retrieval_protocol import validate_request, validate_response

RETRIEVAL_EXECUTION = "shared-coalesced-exact-cache-v4"


class SearchBatcher:
    """One RPC at a time; collect queued requests while that RPC is running.

    The query cap bounds merged requests. A single larger request remains intact;
    DenseRetriever already chunks its encoder/index work into 32-query batches.
    """

    def __init__(self, search, wait_seconds=0.05, max_queries=32, target_queries=None):
        self.search = search
        self.wait_seconds = wait_seconds
        self.max_queries = max_queries
        self.target_queries = target_queries
        self.pending = deque()
        self.worker = None
        self.changed = asyncio.Event()

    async def __call__(self, queries, topk):
        validate_request(queries, topk)
        future = asyncio.get_running_loop().create_future()
        self.pending.append((list(queries), topk, future, asyncio.get_running_loop().time()))
        self.changed.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._drain())
        return await future

    async def close(self):
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        for _, _, future, _ in self.pending:
            if not future.done():
                future.cancel()
        self.pending.clear()

    async def _drain(self):
        dispatch_backlog = False
        while self.pending:
            # The oldest request owns the deadline; later arrivals cannot extend it.
            while self.pending:
                self.changed.clear()
                self.pending = deque(item for item in self.pending if not item[2].cancelled())
                if not self.pending:
                    return
                topk = self.pending[0][1]
                unique = set()
                count = 0
                boundary = False
                for queries, k, _, _ in self.pending:
                    if k != topk or (count and count + len(queries) > self.max_queries):
                        boundary = True
                        break
                    unique.update(queries)
                    count += len(queries)
                remaining = self.pending[0][3] + self.wait_seconds - asyncio.get_running_loop().time()
                if (dispatch_backlog or boundary or remaining <= 0 or
                        (self.target_queries is not None and len(unique) >= self.target_queries)):
                    break
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            selected = []
            count = 0
            topk = None
            while self.pending:
                queries, k, future, arrived = self.pending[0]
                if future.cancelled():
                    self.pending.popleft()
                    continue
                if selected and (k != topk or count + len(queries) > self.max_queries):
                    break
                selected.append(self.pending.popleft())
                topk = k
                count += len(queries)
            if not selected:
                continue
            try:
                dispatched = asyncio.get_running_loop().time()
                combined = [query for queries, _, _, _ in selected for query in queries]
                response = await self.search(combined, topk)
                dispatch_backlog = bool(self.pending)
                validate_response(response, len(combined), topk)
                offset = 0
                metrics_assigned = False
                for queries, _, future, arrived in selected:
                    result = dict(response)
                    result["result"] = response["result"][offset:offset + len(queries)]
                    offset += len(queries)
                    # Count shared backend timings once, not once per HTTP request.
                    result["retrieval_metrics"] = {}
                    if not future.done():
                        if not metrics_assigned:
                            result["retrieval_metrics"] = response.get("retrieval_metrics", {})
                            metrics_assigned = True
                        result["search_batch"] = {"requests": len(selected), "queries": count,
                            "unique_queries": len(set(combined)),
                            "queue_seconds": dispatched - arrived}
                        future.set_result(result)
            except Exception as error:
                # A real backend failure stops this queue and propagates to the runner.
                for _, _, future, _ in [*selected, *self.pending]:
                    if not future.done():
                        future.set_exception(error)
                self.pending.clear()
                return
            except asyncio.CancelledError:
                for _, _, future, _ in [*selected, *self.pending]:
                    if not future.done():
                        future.cancel()
                self.pending.clear()
                raise
