"""Container-wide coalescing; cache/queue run on one asyncio event loop."""
import asyncio
from collections import OrderedDict
from copy import deepcopy
import json

from search_r1_framework.search_batcher import SearchBatcher
from search_r1_framework.retrieval_protocol import validate_request, validate_response


class SharedRetrievalQueue:
    """Merge uncached queries across callers without concurrent index searches.

    The backend is asynchronous and must offload blocking work to one dedicated
    executor. The front cache stays on the event loop, so hits can bypass a busy
    index. Its lifetime is the immutable retriever instance's lifetime.
    """

    def __init__(self, search, *, wait_seconds=0.1, target_queries=4,
                 max_queries=32, cache_entries=2048, log=print):
        self.cache = OrderedDict()
        self.cache_entries = cache_entries
        self.log = log
        self.batch_id = 0

        async def execute(queries, topk):
            self.batch_id += 1
            batch_id = self.batch_id
            started = asyncio.get_running_loop().time()
            response = validate_response(await search(queries, topk), len(queries), topk)
            # Publish the cache before dispatching the next queued batch.
            for query, group in zip(queries, response["result"]):
                key = (query, topk)
                self.cache[key] = deepcopy(group)
                self.cache.move_to_end(key)
                while len(self.cache) > self.cache_entries:
                    self.cache.popitem(last=False)
            response = dict(response, shared_batch_id=batch_id)
            self.log("SHARED_RETRIEVAL_BATCH " + json.dumps({
                "batch_id": batch_id, "queries": len(queries),
                "unique_queries": len(set(queries)), "topk": topk,
                "backend_seconds": asyncio.get_running_loop().time() - started,
                "backend": response.get("retrieval_metrics", {})}))
            return response

        self.batcher = SearchBatcher(execute, wait_seconds, max_queries, target_queries)

    async def __call__(self, queries, topk):
        validate_request(queries, topk)
        started = asyncio.get_running_loop().time()
        found = {}
        missing = []
        for query in dict.fromkeys(queries):
            key = (query, topk)
            if key in self.cache:
                self.cache.move_to_end(key)
                found[query] = deepcopy(self.cache[key])
            else:
                missing.append(query)
        hits = len(found)
        response = {"retrieval_metrics": {}}
        if missing:
            response = await self.batcher(missing, topk)
            found.update(zip(missing, response["result"]))
        response = dict(response)
        response["result"] = [deepcopy(found[q]) for q in queries]
        response["shared_search_batch"] = {
            **response.pop("search_batch", {}),
            "batch_id": response.pop("shared_batch_id", None),
            "front_cache_hit_unique_queries": hits,
            "request_unique_queries": len(found),
            "request_seconds": asyncio.get_running_loop().time() - started,
        }
        self.log("SHARED_RETRIEVAL_REQUEST " + json.dumps(response["shared_search_batch"]))
        return validate_response(response, len(queries), topk)

    async def close(self):
        await self.batcher.close()
