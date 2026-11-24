"""Collect 32 x 8 trajectories, grade rewards, and update the planner once."""
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from peft import LoraConfig, get_peft_model
from transformers import set_seed
from baseline.config import MODEL_ID
from baseline.inference import load_model
from baseline.evaluation import judge_answer
from agent_flow.flow import run_flow
from agent_flow.tools import Toolset
from .policy import PolicyGenerator, update_planner


def judge_rewards(questions, answers, settings, progress=print):
    """Judge independent answers concurrently; assign every result to its rollout."""
    rewards = torch.zeros(len(questions), settings["rollouts_per_question"])
    pool = ThreadPoolExecutor(max_workers=settings.get("reward_workers", 1))
    try:
        pending = {
            pool.submit(judge_answer, question["question"], question["reference_answers"],
                        answer["response"], settings["judge_model"], os.environ["OPENAI_API_KEY"]): (q, r)
            for q, question in enumerate(questions) for r, answer in enumerate(answers[q])
        }
        for done, future in enumerate(as_completed(pending), 1):
            verdict = future.result()
            q, r = pending[future]
            answers[q][r]["reward"] = int(verdict["correct"])
            answers[q][r]["judge_analysis"] = verdict["analysis"]
            rewards[q, r] = answers[q][r]["reward"]
            if done % 32 == 0 or done == len(pending):
                progress(f"Training rewards: {done}/{len(pending)}")
    finally:
        # On failure, cancel requests that have not started and propagate the error.
        pool.shutdown(wait=True, cancel_futures=True)
    return rewards


def load_policy(settings):
    """Initialize one base model and planner adapter for a training session."""
    from importlib.metadata import version

    set_seed(settings["seed"])
    base, tokenizer, versions, _ = load_model(dict(model_id=MODEL_ID, revision=settings["model_revision"]))
    model = get_peft_model(base, LoraConfig(
        r=settings["lora_rank"], lora_alpha=settings["lora_alpha"], lora_dropout=0.0,
        task_type="CAUSAL_LM", target_modules=settings["lora_target_modules"],
    )).eval()
    versions["peft"] = version("peft")
    return model, tokenizer, versions


def train_step(questions, settings, run_python, toolset_class=Toolset,
               model_state=None, optimizer=None):
    started = time.perf_counter()
    stages = []

    def begin(label):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(f"Starting: {label}", flush=True)
        return time.perf_counter()

    def end(label, start):
        torch.cuda.synchronize()
        stages.append(dict(stage=label, seconds=time.perf_counter() - start,
                           peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3))
        print(f"Finished {label}: {stages[-1]['seconds']:.1f}s; "
              f"peak reserved {stages[-1]['peak_gpu_reserved_gib']:.2f} GiB", flush=True)

    if model_state is None:
        phase = begin("model_load")
        model_state = load_policy(settings)
        end("model_load", phase)
    model, tokenizer, versions = model_state

    phase = begin("rollout")
    count = settings["rollouts_per_question"]
    samples = [[[] for _ in range(count)] for _ in questions]
    answers = [[None for _ in range(count)] for _ in questions]
    tasks = [(q, r) for q in range(len(questions)) for r in range(count)]
    generate = PolicyGenerator(model, tokenizer, settings["rollout_batch_size"],
                               reuse_identical_requests=settings.get("reuse_identical_requests", False))
    tools = toolset_class(generate, settings, run_python)
    execute_tools = getattr(tools, "execute_many", None)
    for offset in range(0, len(tasks), settings["rollout_batch_size"]):
        batch = tasks[offset:offset + settings["rollout_batch_size"]]

        def record_planner(index, generated):
            q, r = batch[index]
            samples[q][r].append({key: generated[key] for key in ("prompt_ids", "action_ids")})

        states = run_flow([questions[q]["question"] for q, _ in batch], settings,
                          generate, tools.execute, record_planner=record_planner,
                          execute_tools=execute_tools)
        for (q, r), state in zip(batch, states):
            answers[q][r] = dict(response=state["response"], turns=len(state["trajectory"]),
                                 stop_reason=state["stop_reason"], finish_reason=state["finish_reason"])
        del states
        # Histories and token IDs stay on CPU; no generation cache is kept.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Collected trajectories: {offset + len(batch)}/{len(tasks)}", flush=True)
    end("rollout", phase)
    if execute_tools is not None:
        stages[-1]["qwen_generation_seconds"] = generate.generator.seconds
    if settings.get("reuse_identical_requests", False):
        stages[-1]["qwen_requests_reused"] = generate.reused_requests

    phase = begin("reward_judging")
    rewards = judge_rewards(questions, answers, settings,
                            progress=lambda message: print(message, flush=True))
    end("reward_judging", phase)

    phase = begin("parameter_update")
    metrics = update_planner(model, samples, rewards, settings,
                             progress=lambda message: print(message, flush=True), optimizer=optimizer)
    end("parameter_update", phase)
    metrics["mean_training_reward"] = float(rewards.mean())
    return dict(gpu=torch.cuda.get_device_name(),
                gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
                versions=versions, stages=stages,
                server_total_seconds=time.perf_counter() - started,
                peak_gpu_reserved_gib=max(s["peak_gpu_reserved_gib"] for s in stages),
                update=metrics,
                questions=[dict(question, rollouts=group) for question, group in zip(questions, answers)])
