"""Translate explicit assignment settings into veRL v0.6.1 overrides."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from search_r1_framework.search_wait import SEARCH_HTTP_TIMEOUT_SECONDS

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
VERL_VERSION = "0.6.1"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# veRL 0.6.1 HFModelConfig accepts a string, while PEFT also accepts a list.
# Preserve PEFT's exact-name or dotted-suffix matching for these seven modules.
# Use [.] rather than a backslash escape so Hydra preserves the regex in CLI overrides.
TARGET_MODULES_PATTERN = "(?:.*[.])?(?:" + "|".join(TARGET_MODULES) + ")"


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    train_questions: int = 2000
    dev_questions: int = 200
    # Preserve the prepared pool; sample this many questions for training validation.
    validation_questions: int = 10
    steps: int = 100
    questions_per_step: int = 1
    responses_per_question: int = 8
    micro_batch_per_gpu: int = 8
    learning_rate: float = 2e-5
    critic_learning_rate: float = 1e-5
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    beta: float = 0.001
    prompt_tokens: int = 384
    response_tokens: int = 2560
    assistant_turns: int = 5
    # veRL 0.6.1 truncates tool response text by characters, before tokenization.
    tool_response_chars: int = 512
    vllm_memory_fraction: float = 0.3
    eval_every: int = 100
    save_every: int = 10

    def validate(self):
        counts = (self.train_questions, self.dev_questions, self.validation_questions, self.steps, self.questions_per_step,
                  self.micro_batch_per_gpu, self.prompt_tokens, self.response_tokens,
                  self.assistant_turns, self.tool_response_chars, self.eval_every, self.save_every)
        if any(type(x) is not int or x <= 0 for x in counts):
            raise ValueError("Counts must be positive integers.")
        if self.train_questions % 2 or self.dev_questions % 2:
            raise ValueError("Use even counts for equal NQ/HotpotQA sampling.")
        if self.responses_per_question < 2 or self.rank not in (8, 16, 32, 64):
            raise ValueError("GRPO requires at least two responses; choose a vLLM-supported LoRA rank.")
        if self.train_questions < self.questions_per_step:
            raise ValueError("Not enough training questions for one batch.")
        if (self.questions_per_step * self.responses_per_question) % self.micro_batch_per_gpu:
            raise ValueError("The response batch must be divisible by the micro-batch size.")
        if not (0 < self.vllm_memory_fraction < 1 and 0 <= self.dropout < 1):
            raise ValueError("Invalid memory fraction or dropout.")
        if min(self.learning_rate, self.critic_learning_rate, self.alpha) <= 0 or self.beta < 0:
            raise ValueError("Invalid learning rate, alpha, or KL coefficient.")
        return self

    def to_dict(self):
        return asdict(self)


def tool_config():
    return {"tools": [{
        "class_name": "verl.tools.search_tool.SearchTool",
        "config": {"retrieval_service_url": "http://127.0.0.1:8000/retrieve", "topk": 3,
                   "num_workers": 8, "rate_limit": 8, "timeout": SEARCH_HTTP_TIMEOUT_SECONDS, "type": "native"},
        "tool_schema": {"type": "function", "function": {
            "name": "search", "description": "Search Wikipedia passages for factual evidence.",
            "parameters": {"type": "object", "properties": {
                # veRL 0.6.1 drops items/minItems/maxItems during schema loading.
                # State the element contract in a supported field and validate at the server.
                "query_list": {"type": "array",
                               "description": 'A nonempty list of search strings, e.g. ["author of book", "author birthplace"]. Even one query must be a list: ["query text"], never a plain string.'}},
                "required": ["query_list"]}}}}]}


def write_support_configs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # JSON is also valid YAML; OmegaConf loads these files directly.
    (directory / "tools.yaml").write_text(json.dumps(tool_config(), indent=2), encoding="utf-8")
    agents = [
        {"name": "assignment_search", "_target_": "search_r1_framework.agents.ScoredSearchAgent"},
        {"name": "assignment_direct", "_target_": "search_r1_framework.agents.ScoredDirectAgent"},
    ]
    (directory / "agents.yaml").write_text(json.dumps(agents, indent=2), encoding="utf-8")


def overrides(settings, algorithm, prepared, output, model_path, actor_adapter=None,
              critic_adapter=None, evaluation=False, direct=False, resume=False, resume_checkpoint=None):
    """Use framework configuration only; PPO/GRPO loss and updates remain in veRL."""
    s = settings.validate()
    if resume and (evaluation or not resume_checkpoint):
        raise ValueError("Resume requires an explicitly verified training checkpoint path.")
    if algorithm not in ("ppo", "grpo") or (direct and not evaluation):
        raise ValueError("Expected PPO/GRPO, or a direct inference evaluation.")
    p, o = Path(prepared), Path(output)
    search = not direct
    critic = algorithm == "ppo" and not evaluation
    if not direct and not actor_adapter:
        raise ValueError("Provide the initialized or trained actor adapter.")
    if critic and not critic_adapter:
        raise ValueError("PPO needs the initialized token-classification critic adapter.")
    values = {
        "algorithm.adv_estimator": "gae" if critic else "grpo",
        "algorithm.gamma": 1.0, "algorithm.lam": 1.0,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(p / "train.parquet"),
        "data.val_files": str(p / ("bamboogle_direct.parquet" if direct else
                                  "bamboogle_search.parquet" if evaluation else "dev.parquet")),
        "data.train_batch_size": s.questions_per_step,
        "data.val_batch_size": 16,
        # Final Bamboogle evaluation must use every question, not the development cap.
        "data.val_max_samples": -1 if evaluation else min(s.validation_questions, s.dev_questions),
        "data.max_prompt_length": s.prompt_tokens,
        "data.max_response_length": s.response_tokens,
        "data.return_raw_chat": True, "data.shuffle": True, "data.seed": s.seed,
        "data.validation_shuffle": False, "data.dataloader_num_workers": 0,
        "data.filter_overlong_prompts": False, "data.truncation": "error",
        "actor_rollout_ref.model.path": str(model_path),
        "actor_rollout_ref.model.lora_rank": 0 if direct else s.rank,
        "actor_rollout_ref.model.lora_alpha": s.alpha,
        "actor_rollout_ref.model.target_modules": TARGET_MODULES_PATTERN,
        "actor_rollout_ref.model.lora_adapter_path": actor_adapter,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": False,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": s.learning_rate,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.ppo_mini_batch_size": s.questions_per_step,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": s.micro_batch_per_gpu,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.use_kl_loss": not evaluation,
        "actor_rollout_ref.actor.kl_loss_coef": s.beta,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": s.micro_batch_per_gpu,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.n": s.responses_per_question,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.gpu_memory_utilization": s.vllm_memory_fraction,
        "actor_rollout_ref.rollout.max_num_seqs": 32,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        # veRL 0.6.1 stringifies list-valued CLI options incorrectly. Pass the
        # same capture sizes through vLLM's native JSON compilation configuration.
        "actor_rollout_ref.rollout.cudagraph_capture_sizes": None,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config": json.dumps(
            {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}),
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": s.micro_batch_per_gpu,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.agent.num_workers": 4,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "assignment_search" if search else "assignment_direct",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(o / "runtime_config" / "agents.yaml"),
        "actor_rollout_ref.rollout.multi_turn.enable": search,
        "actor_rollout_ref.rollout.multi_turn.tool_config_path": str(o / "runtime_config" / "tools.yaml") if search else None,
        "actor_rollout_ref.rollout.multi_turn.format": "hermes",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": s.assistant_turns,
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls": 1,
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length": s.tool_response_chars,
        "critic.enable": critic, "critic.strategy": "fsdp",
        "critic.model.path": str(model_path),
        "critic.model.lora_rank": s.rank, "critic.model.lora_alpha": s.alpha,
        "critic.model.target_modules": TARGET_MODULES_PATTERN,
        "critic.model._target_": "search_r1_framework.compat.CriticModelConfig",
        "+critic.model.lora_adapter_path": critic_adapter,
        "critic.model.fsdp_config.model_dtype": "bfloat16",
        "critic.model.fsdp_config.param_offload": True,
        "critic.model.fsdp_config.optimizer_offload": True,
        "critic.model.use_remove_padding": True,
        "critic.model.enable_gradient_checkpointing": False,
        "critic.optim.lr": s.critic_learning_rate,
        "critic.ppo_micro_batch_size_per_gpu": s.micro_batch_per_gpu,
        "critic.forward_micro_batch_size_per_gpu": s.micro_batch_per_gpu,
        "trainer.use_legacy_worker_impl": "enable",
        "trainer.project_name": "week08-search-r1-framework",
        "trainer.experiment_name": o.name,
        "trainer.n_gpus_per_node": 1, "trainer.nnodes": 1,
        "trainer.total_training_steps": s.steps,
        # Enough epochs to reach the explicit step cap even on a small dataset.
        "trainer.total_epochs": s.steps,
        "trainer.val_before_train": True, "trainer.val_only": evaluation,
        "trainer.test_freq": s.eval_every, "trainer.save_freq": s.save_every,
        "trainer.logger": ["console", "tensorboard"],
        "trainer.log_val_generations": 0,
        "trainer.rollout_data_dir": None,
        "trainer.validation_data_dir": str(o / "evaluation") if evaluation else None,
        "trainer.default_local_dir": str(o / "checkpoints"),
        "trainer.resume_mode": "resume_path" if resume else "disable",
        "trainer.resume_from_path": str(resume_checkpoint) if resume else None,
        "trainer.max_actor_ckpt_to_keep": 2, "trainer.max_critic_ckpt_to_keep": 2,
        "ray_kwargs.ray_init.num_cpus": 8,
    }
    return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]
