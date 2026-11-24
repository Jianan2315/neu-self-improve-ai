"""H200 entry point for the one-step training test."""
from functools import partial
import modal
from agent_flow.modal_app import base_image, cache, run_python

image = base_image.pip_install("peft==0.17.1")
if modal.is_local():
    image = image.add_local_python_source("baseline", "agent_flow", "flow_grpo")
app = modal.App("week09-flow-grpo-step-test")


@app.function(image=image, gpu="H200", cpu=4, memory=65536,
              volumes={"/cache": cache}, timeout=86400, startup_timeout=1800,
              scaledown_window=60, max_containers=1, retries=0)
def training_step(questions: list[dict], settings: dict):
    from .training import train_step
    result = train_step(questions, settings, partial(run_python, sandbox_app=app))
    cache.commit()
    return result
