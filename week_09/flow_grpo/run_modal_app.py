"""Modal entry point for ten training updates and the final benchmark evaluation."""
from functools import partial
import modal
from agent_flow.modal_app import base_image, cache, run_python

image = base_image.pip_install("peft==0.17.1")
if modal.is_local():
    image = image.add_local_python_source("baseline", "agent_flow", "flow_grpo")
checkpoints = modal.Volume.from_name("week09-flow-grpo-checkpoints", create_if_missing=True)
app = modal.App("week09-flow-grpo")


def train_and_evaluate(run_name: str, batches: list, settings: dict,
                       benchmark_questions: list[str], evaluation_settings: dict,
                       resume: bool = False):
    from flow_grpo.run import run_training
    return run_training(run_name, batches, settings, benchmark_questions, evaluation_settings,
                        partial(run_python, sandbox_app=app), "/checkpoints", checkpoints.commit, resume)


def create_app(gpu, tool_secret):
    """Register the training function with the notebook's GPU selection."""
    remote_function = app.function(
        name="train_and_evaluate", image=image, gpu=gpu, secrets=[tool_secret],
        cpu=4, memory=65536, volumes={"/cache": cache, "/checkpoints": checkpoints},
        timeout=86400, startup_timeout=1800, scaledown_window=2,
        max_containers=1, retries=0,
    )(train_and_evaluate)
    return app, remote_function
