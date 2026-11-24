"""Modal GPU entry point and isolated CPU execution for Python Coder."""
import modal
from .inference import run_trials

cache = modal.Volume.from_name("week09-model-cache", create_if_missing=True)
base_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.10.0", setup_dockerfile_commands=[
        "RUN test -x /usr/bin/python3.12 && ln -sf /usr/bin/python3.12 /usr/local/bin/python",
        "RUN python --version && python -m pip --version",
    ]).entrypoint([])
    .pip_install("transformers==4.55.4", "accelerate==1.10.1")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("beautifulsoup4==4.13.4", "requests==2.32.4")
)
image = base_image
if modal.is_local():
    image = image.add_local_python_source("baseline", "agent_flow")
python_image = modal.Image.debian_slim(python_version="3.12")
app = modal.App("week09-agentflow")


def run_python(code, sandbox_app=None):
    # Generated code runs without the GPU container's secrets or cache volume.
    wrapper = """import resource
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 ** 2, 256 * 1024 ** 2))
remaining = 8000
def limited_print(*values, sep=' ', end='\\n'):
    global remaining
    text = sep.join(str(v) for v in values) + end
    if len(text) > remaining:
        raise ValueError('Python output limit exceeded')
    remaining -= len(text)
    print(text, end='')
scope = {'print': limited_print}
exec(CODE, scope)
""".replace("CODE", repr(code))
    sandbox = modal.Sandbox.create("python", "-I", "-c", wrapper, app=sandbox_app or app,
                                   image=python_image, timeout=15, cpu=1, memory=512,
                                   block_network=True)
    try:
        exit_code = sandbox.wait()
        text, error = sandbox.stdout.read(), sandbox.stderr.read()
        if exit_code:
            return {"error": error[-2000:] or "Python calculation exceeded its limits."}
        return {"text": text.strip()}
    finally:
        sandbox.terminate()


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768,
              volumes={"/cache": cache}, timeout=14400, startup_timeout=1800,
              scaledown_window=60, max_containers=1, retries=0)
def answer_trials(questions: list[str], settings: dict):
    result = run_trials(questions, settings, run_python)
    cache.commit()
    return result
