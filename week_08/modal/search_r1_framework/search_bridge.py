"""HTTP transport only; search execution and tool feedback remain in veRL."""
import asyncio
import json
import time
from pathlib import Path
from search_r1_framework.retrieval_protocol import validate_request, validate_response
from search_r1_framework.search_wait import SEARCH_BRIDGE_TIMEOUT_SECONDS


def create_search_api(search, failures, log_path=None, timeout=SEARCH_BRIDGE_TIMEOUT_SECONDS):
    from fastapi import FastAPI, HTTPException
    api = FastAPI()

    def record(kind, payload, detail):
        entry = {"time": time.time(), "kind": kind, "request": payload, "detail": detail}
        print("SEARCH_BRIDGE " + json.dumps(entry, ensure_ascii=False), flush=True)
        if log_path:
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    api.state.record_search_event = record

    @api.post("/retrieve")
    async def retrieve(payload: dict):
        queries, topk = payload.get("queries"), payload.get("topk", 3)
        try:
            validate_request(queries, topk)
            if payload.get("return_scores", True) is not True:
                raise ValueError("This endpoint requires return_scores=true.")
        except ValueError as error:
            record("invalid_request", payload, str(error))
            # veRL does not retry 4xx. Never coerce strings into query lists.
            raise HTTPException(status_code=400, detail=str(error)) from error
        if failures:
            raise HTTPException(status_code=503, detail="Search backend failed; runner is stopping.")
        try:
            started = time.monotonic()
            response = await asyncio.wait_for(search(queries, topk), timeout=timeout)
            validate_response(response, len(queries), topk)
            record("request_completed", {"query_count": len(queries), "topk": topk},
                   {"rpc_wait_seconds": time.monotonic() - started,
                    "backend": response.get("retrieval_metrics", {}),
                    "search_batch": response.get("search_batch", {}),
                    "shared_search_batch": response.get("shared_search_batch", {})})
            return response
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            failures.append(detail)
            record("backend_failure", payload, detail)
            raise HTTPException(status_code=503, detail="Search backend failed; runner is stopping.") from error

    return api
