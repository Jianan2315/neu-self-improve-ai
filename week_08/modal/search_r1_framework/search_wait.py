"""Bounded result polling for one Modal call; polling never resubmits work."""
import asyncio
import time

SEARCH_RESULT_TIMEOUT_SECONDS = 600
SEARCH_BRIDGE_TIMEOUT_SECONDS = 620
SEARCH_HTTP_TIMEOUT_SECONDS = 650
SEARCH_POLL_SECONDS = 30
SEARCH_EXTENDED_WAIT_SECONDS = 175
SEARCH_CANCEL_TIMEOUT_SECONDS = 10


def poll_function_call(call, notify, *, poll_seconds=SEARCH_POLL_SECONDS):
    """Poll the same SDK call; builtin TimeoutError means no result yet.

    Modal 1.6.4's _Invocation.poll_function raises builtin TimeoutError for an
    empty poll. modal.exception.TimeoutError instead denotes a remote execution
    failure and must propagate. The caller owns cancellation and cleanup.
    """
    while True:
        try:
            return call.get(timeout=poll_seconds)
        except TimeoutError:
            notify()


async def wait_for_search_call(call, timeout_error, notify, *,
                               total_timeout=SEARCH_RESULT_TIMEOUT_SECONDS,
                               poll_seconds=SEARCH_POLL_SECONDS,
                               extended_after=SEARCH_EXTENDED_WAIT_SECONDS,
                               clock=time.monotonic):
    started = clock()
    extended = False
    try:
        while True:
            elapsed = clock() - started
            remaining = total_timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    f"Search call {call.object_id} did not return within {total_timeout}s "
                    "including queueing and possible container recovery.")
            try:
                # A polling timeout says only that this same call has no result yet.
                result = await call.get.aio(timeout=min(poll_seconds, remaining))
            except timeout_error:
                elapsed = clock() - started
                if elapsed >= extended_after and not extended:
                    extended = True
                    notify("extended_wait", {"call_id": call.object_id, "elapsed_seconds": elapsed,
                                             "reason": "Pending result; queueing or restart is not yet distinguished."})
                continue
            if extended:
                notify("request_recovered", {"call_id": call.object_id, "elapsed_seconds": clock() - started})
            return result
    except BaseException:
        try:
            await asyncio.wait_for(call.cancel.aio(), timeout=SEARCH_CANCEL_TIMEOUT_SECONDS)
        except Exception as error:
            notify("cancellation_failed", {"call_id": call.object_id, "error": repr(error)})
        raise


class RetrieverInstanceTracker:
    """Detect replacement of the single configured retriever container."""
    def __init__(self, initial_id):
        self.current_id = initial_id

    def observe(self, response, notify):
        instance = response.get("retriever_instance_id") if isinstance(response, dict) else None
        if not isinstance(instance, str) or not instance:
            raise RuntimeError("Missing retriever instance ID; redeploy both sides of the search service.")
        if self.current_id != instance:
            notify("retriever_restarted", {"previous_instance": self.current_id, "current_instance": instance})
            self.current_id = instance
