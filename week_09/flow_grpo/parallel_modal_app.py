"""H200 entry point for the concurrent-tool one-step training test."""
from functools import partial
import modal
from agent_flow.modal_app import run_python
from .modal_app import image, cache

app = modal.App("week09-flow-grpo-parallel-test")


@app.function(image=image, gpu="H200", cpu=4, memory=65536,
              volumes={"/cache": cache}, timeout=86400, startup_timeout=1800,
              scaledown_window=60, max_containers=1, retries=0)
def training_step(questions: list[dict], settings: dict):
    from .training import train_step
    from .parallel_tools import ConcurrentToolset
    result = train_step(questions, settings, partial(run_python, sandbox_app=app),
                        toolset_class=ConcurrentToolset)
    cache.commit()
    return result
