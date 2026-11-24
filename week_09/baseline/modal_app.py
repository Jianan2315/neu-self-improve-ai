"""Modal entry point for sequential baseline trials."""
import modal
from .config import MODEL_ID
from .inference import load_model, infer, run_sequential

cache = modal.Volume.from_name("week09-model-cache", create_if_missing=True)

# Retain the validated CUDA and inference-library versions.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.10.0", setup_dockerfile_commands=[
        "RUN test -x /usr/bin/python3.12 && ln -sf /usr/bin/python3.12 /usr/local/bin/python",
        "RUN python --version && python -m pip --version",
    ]).entrypoint([])
    .pip_install("transformers==4.55.4", "accelerate==1.10.1")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
)

if modal.is_local():
    image = image.add_local_python_source("baseline")

app = modal.App("week09-qwen7b-sequential-baseline")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768,
              volumes={"/cache": cache}, timeout=1800, startup_timeout=1800,
              scaledown_window=60, max_containers=1, retries=0)
def answer_trials(questions: list[str], seeds: list[int], revision: str,
                  temperature: float, max_new_tokens: int):
    if not 1 <= len(questions) <= 125 or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Expected 1 to 125 nonempty questions.")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers.")
    if not 0 <= temperature <= 2 or not 1 <= max_new_tokens <= 2048:
        raise ValueError("Unsupported generation settings.")
    jobs = [dict(questions=questions, seed=seed, revision=revision, model_id=MODEL_ID,
                 temperature=temperature, max_new_tokens=max_new_tokens) for seed in seeds]
    result = run_sequential(jobs, load_model, infer)
    cache.commit()
    return result
