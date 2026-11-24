"""Model loading, batched answer generation and sequential GPU trials."""
import time
from .config import SYSTEM_PROMPT


def load_model(job):
    import re
    from importlib.metadata import version
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not re.fullmatch(r"[0-9a-f]{40}", job["revision"]):
        raise ValueError("An immutable model commit is required.")
    expected = {"torch": "2.7.1", "transformers": "4.55.4", "accelerate": "1.10.1"}
    versions = {name: version(name) for name in expected}
    if any(versions[name].split("+")[0] != wanted for name, wanted in expected.items()):
        raise RuntimeError(f"Unexpected cloud library versions: {versions}")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(job["model_id"], revision=job["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        job["model_id"], revision=job["revision"], torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    return model, tokenizer, versions, generate_batch


def infer(state, job):
    import torch
    model, tokenizer, versions, generate_batch = state
    result = generate_batch(model, tokenizer, job["questions"], job["seed"],
                            job["temperature"], job["max_new_tokens"])
    result.update(gpu=torch.cuda.get_device_name(), versions=versions)
    return result


def trim_generated_tokens(tokens, eos_ids):
    """Include the first EOS in the count; exclude subsequent batch padding."""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1], 'eos'
    return tokens, 'length'


def generate_batch(model, tokenizer, questions, seed, temperature, max_new_tokens):
    """Generate one batch on the container's GPU."""
    import gc
    import time
    import torch
    from transformers import GenerationConfig, set_seed
    inputs = sequences = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = {'seed': seed}
    try:
        set_seed(seed)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': question}],
            tokenize=False, add_generation_prompt=True,
        ) for question in questions]
        inputs = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda:0')
        padded_input_width = inputs.input_ids.shape[1]
        options = dict(
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            repetition_penalty=1.0, use_cache=True,
        )
        if temperature > 0:
            options.update(temperature=temperature, top_p=1.0, top_k=0)
        generation = GenerationConfig(**options)
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(**inputs, generation_config=generation)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        eos = generation.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        responses = []
        for index, sequence in enumerate(sequences):
            tokens, reason = trim_generated_tokens(
                sequence[padded_input_width:].tolist(), eos_ids)
            responses.append({
                'question': questions[index],
                'response': tokenizer.decode(tokens, skip_special_tokens=True),
                'input_tokens': int(inputs.attention_mask[index].sum()),
                'output_tokens': len(tokens), 'finish_reason': reason,
            })
        output_tokens = sum(row['output_tokens'] for row in responses)
        result.update(
            responses=responses, padded_input_tokens_per_row=padded_input_width,
            total_input_tokens=sum(row['input_tokens'] for row in responses),
            total_output_tokens=output_tokens, generation_seconds=generation_seconds,
            questions_per_generation_second=len(questions) / generation_seconds,
            output_tokens_per_generation_second=output_tokens / generation_seconds,
        )
        result.update(
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )
    finally:
        inputs = sequences = None
        gc.collect()
        torch.cuda.empty_cache()
    return result


def run_sequential(jobs, load, infer):
    if not jobs or len({j["seed"] for j in jobs}) != len(jobs):
        raise ValueError("Expected nonempty, distinct seed jobs.")
    started = time.perf_counter()
    outputs = []
    shared = {}
    state = load(jobs[0])
    loaded = time.perf_counter()
    for job in jobs:
        result = infer(state, job)
        for key in ("gpu", "gpu_total_gib", "versions", "padded_input_tokens_per_row", "total_input_tokens"):
            value = result.pop(key)
            if not outputs:
                shared[key] = value
        outputs.append(result)
    finished = time.perf_counter()
    return {
        **shared, "outputs": outputs,
        "model_load_seconds": loaded - started,
        "trial_loop_seconds": finished - loaded,
        "server_total_seconds": finished - started,
    }
