"""Persistent multi-step training, latest checkpoint, and trained-planner evaluation."""
import gc
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import set_seed
from agent_flow.flow import run_flow
from .parallel_tools import ConcurrentToolset
from .policy import PolicyGenerator
from .training import load_policy, train_step


def save_checkpoint(path, state, model, optimizer, commit):
    """Replace the latest complete checkpoint; keep weights and history together."""
    payload = dict(state, adapter={k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                   optimizer=optimizer.state_dict(),
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    commit()


def evaluate_trial(model, tokenizer, questions, settings, seed, run_python):
    """Evaluate one seed with the trained planner and frozen remaining modules."""
    set_seed(seed)
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generate = PolicyGenerator(model, tokenizer, settings["batch_size"],
                               reuse_identical_requests=settings["reuse_identical_requests"])
    tools = ConcurrentToolset(generate, settings, run_python)
    answers = []
    for offset in range(0, len(questions), settings["batch_size"]):
        batch = questions[offset:offset + settings["batch_size"]]
        answers.extend(run_flow(batch, settings, generate, tools.execute, execute_tools=tools.execute_many))
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Evaluation seed {seed}: {len(answers)}/{len(questions)} answers", flush=True)
    torch.cuda.synchronize()
    return dict(seed=seed, responses=answers, trial_seconds=time.perf_counter() - started,
                generation_seconds=generate.generator.seconds,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)


def run_training(run_name, batches, settings, benchmark_questions, evaluation_settings,
                 run_python, checkpoint_root, commit, resume=False):
    """Train sequential updates with one model/optimizer, then evaluate three seeds."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscores or hyphens.")
    directory = Path(checkpoint_root) / run_name
    checkpoint_path = directory / "checkpoint.pt"
    if resume:
        # This file is produced by this training program, not an external checkpoint.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings or state["evaluation_settings"] != evaluation_settings:
            raise ValueError("Resume with the saved training and evaluation settings.")
        if state["benchmark_questions"] != benchmark_questions:
            raise ValueError("Resume with the same benchmark questions.")
        batches = state["batches"]
    else:
        directory.mkdir(parents=True, exist_ok=False)
        if len(batches) != settings["training_steps"]:
            raise ValueError("Provide one training batch for each requested update.")
        state = dict(settings=settings, evaluation_settings=evaluation_settings, batches=batches,
                     benchmark_questions=benchmark_questions,
                     training=dict(steps=[]), evaluation=dict(outputs=[]))

    model_state = load_policy(settings)
    model, tokenizer, versions = model_state
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=settings["learning_rate"], weight_decay=0.0)
    if resume:
        set_peft_model_state_dict(model, state.pop("adapter"))
        optimizer.load_state_dict(state.pop("optimizer"))
        rng = state.pop("rng")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"Resumed after {len(state['training']['steps'])} completed updates", flush=True)

    hardware = dict(gpu=torch.cuda.get_device_name(),
                    gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
                    versions=versions)
    state["training"].update(hardware)
    state["evaluation"].update(hardware)
    for step in range(len(state["training"]["steps"]), settings["training_steps"]):
        print(f"Training update {step + 1}/{settings['training_steps']}", flush=True)
        output = train_step(batches[step], settings, run_python, toolset_class=ConcurrentToolset,
                            model_state=model_state, optimizer=optimizer)
        for key in hardware:
            output.pop(key)
        state["training"]["steps"].append(dict(step=step + 1, **output))
        save_checkpoint(checkpoint_path, state, model, optimizer, commit)
        print(f"Saved checkpoint after update {step + 1}", flush=True)

    # Evaluation does not need AdamW's GPU buffers. Keep them on CPU in checkpoints.
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[name] = value.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    for seed in evaluation_settings["seeds"][len(state["evaluation"]["outputs"]):]:
        trial = evaluate_trial(model, tokenizer, benchmark_questions, evaluation_settings, seed, run_python)
        state["evaluation"]["outputs"].append(trial)
        save_checkpoint(checkpoint_path, state, model, optimizer, commit)
        print(f"Saved evaluation seed {seed}", flush=True)
    return dict(training=state["training"], evaluation=state["evaluation"])
