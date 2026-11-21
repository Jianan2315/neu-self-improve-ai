"""Strict shared contract for veRL SearchTool, the HTTP bridge, and Modal RPC."""
import math


def validate_request(queries, topk):
    if not isinstance(queries, list) or not queries:
        raise ValueError('queries must be a nonempty JSON array, for example ["query text"].')
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Every query must be a nonempty string.")
    if type(topk) is not int or topk < 1:
        raise ValueError("topk must be a positive integer.")


def validate_response(response, query_count, topk):
    if not isinstance(response, dict) or not isinstance(response.get("result"), list):
        raise ValueError("Retriever response must contain a result array.")
    if len(response["result"]) != query_count:
        raise ValueError("Retriever must return one result list per query, in request order.")
    for group in response["result"]:
        if not isinstance(group, list) or len(group) > topk:
            raise ValueError("Each result group must be a list with at most topk passages.")
        for item in group:
            document = item.get("document") if isinstance(item, dict) else None
            if not isinstance(document, dict) or not isinstance(document.get("contents"), str):
                raise ValueError("Each retrieved passage must have document.contents text.")
            score = item.get("score")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("Each retrieved passage must have a finite numeric score.")
    return response


def retrieve_batch(engine, queries, topk):
    validate_request(queries, topk)
    try:
        response = {"result": [
            [{"document": {"id": item["id"], "contents": item["contents"]}, "score": item["score"]}
             for item in group] for group in engine.retrieve_many(queries, topk)]}
        if hasattr(engine, "last_batch_metrics"):
            response["retrieval_metrics"] = engine.last_batch_metrics
    except TimeoutError as error:
        # Modal get(timeout=...) uses the same built-in exception for a pending
        # result. Keep a completed backend failure distinguishable at that boundary.
        raise RuntimeError(f"Retriever execution raised TimeoutError: {error}") from error
    return validate_response(response, len(queries), topk)
