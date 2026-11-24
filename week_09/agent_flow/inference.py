"""One frozen Qwen model serves all AgentFlow roles; trials run sequentially."""
import time
from baseline.config import MODEL_ID
from baseline.inference import load_model, trim_generated_tokens
from .flow import run_flow
from .tools import Toolset


class Generator:
    def __init__(self, model, tokenizer, batch_size, capture_tokens=False):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        self.capture_tokens = capture_tokens
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.input_tokens = self.output_tokens = 0
        self.seconds = 0.0

    def __call__(self, conversations, temperature, max_new_tokens):
        import torch
        from transformers import GenerationConfig
        rows = []
        for offset in range(0, len(conversations), self.batch_size):
            prompts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                       for c in conversations[offset:offset + self.batch_size]]
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:0")
            width = inputs.input_ids.shape[1]
            if width + max_new_tokens > self.model.config.max_position_embeddings:
                raise ValueError("AgentFlow memory exceeds the model context window.")
            options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                           pad_token_id=self.tokenizer.eos_token_id,
                           eos_token_id=self.model.generation_config.eos_token_id, use_cache=True)
            if temperature > 0:
                options.update(temperature=temperature, top_p=1.0, top_k=0)
            config = GenerationConfig(**options)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                sequences = self.model.generate(**inputs, generation_config=config)
            torch.cuda.synchronize()
            self.seconds += time.perf_counter() - started
            eos = config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            for i, sequence in enumerate(sequences):
                tokens, reason = trim_generated_tokens(sequence[width:].tolist(), eos_ids)
                row = dict(text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                           input_tokens=int(inputs.attention_mask[i].sum()), output_tokens=len(tokens),
                           finish_reason=reason)
                if self.capture_tokens:
                    row["prompt_ids"] = inputs.input_ids[i][inputs.attention_mask[i].bool()].tolist()
                    row["action_ids"] = tokens
                self.input_tokens += row["input_tokens"]
                self.output_tokens += row["output_tokens"]
                rows.append(row)
            del inputs, sequences, sequence
        return rows


def run_trials(questions, settings, run_python):
    import gc
    import torch
    from transformers import set_seed
    started = time.perf_counter()
    model, tokenizer, versions, _ = load_model(dict(model_id=MODEL_ID, revision=settings["model_revision"]))
    loaded = time.perf_counter()
    outputs = []
    for seed in settings["seeds"]:
        set_seed(seed)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generate = Generator(model, tokenizer, settings["batch_size"])
        tools = Toolset(generate, settings, run_python)
        trial_started = time.perf_counter()
        answers = run_flow(questions, settings, generate, tools.execute)
        outputs.append(dict(seed=seed, responses=answers,
                            trial_seconds=time.perf_counter() - trial_started,
                            generation_seconds=generate.seconds,
                            total_input_tokens=generate.input_tokens,
                            total_output_tokens=generate.output_tokens,
                            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3))
    finished = time.perf_counter()
    return dict(gpu=torch.cuda.get_device_name(),
                gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
                versions=versions, outputs=outputs, model_load_seconds=loaded - started,
                trial_loop_seconds=finished - loaded, server_total_seconds=finished - started)
