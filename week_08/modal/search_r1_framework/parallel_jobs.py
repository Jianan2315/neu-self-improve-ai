"""Observe already submitted jobs without resubmission or implicit cancellation."""
import time


def wait_for_calls(calls, notify=print, interval_seconds=15):
    """Calls are Modal FunctionCall objects; each poll is non-blocking.

    A real failure stops the local workflow. Other submitted jobs remain live,
    and their IDs are reported for inspection rather than silently cancelled.
    """
    pending = dict(calls)
    completed = {}
    while pending:
        failures = {}
        for label, call in list(pending.items()):
            try:
                completed[label] = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as error:
                failures[label] = str(error)
                pending.pop(label)
                notify(f"FAILED {label}: {error}")
            else:
                pending.pop(label)
                notify(f"COMPLETED {label}")
        if failures:
            live = {label: call.object_id for label, call in pending.items()}
            raise RuntimeError(f"Cloud jobs failed: {failures}. Other pending calls were not cancelled: {live}. "
                               "Inspect saved call IDs and logs before any retry.")
        if pending:
            notify("WAITING " + ", ".join(f"{label} ({call.object_id})" for label, call in pending.items()))
            time.sleep(interval_seconds)
    return completed
