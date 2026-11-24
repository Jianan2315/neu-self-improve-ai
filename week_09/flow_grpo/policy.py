"""One base model, a planner adapter, and exact per-turn training inputs."""
from contextlib import nullcontext
import json
import torch
from torch.utils.checkpoint import checkpoint
from agent_flow.inference import Generator
from agent_flow.prompts import PROMPTS


class PolicyGenerator:
    def __init__(self, model, tokenizer, batch_size, reuse_identical_requests=False):
        self.model = model
        self.generator = Generator(model, tokenizer, batch_size)
        self.reuse_identical_requests = reuse_identical_requests
        self.reused_requests = 0

    def __call__(self, conversations, temperature, max_new_tokens):
        if not conversations:
            return []
        planner = conversations[0][0]["content"] == PROMPTS["planner"]
        self.generator.capture_tokens = planner
        # Sample every planner trajectory independently. Reuse only greedy,
        # frozen-model requests with identical complete inputs in this call.
        reuse = self.reuse_identical_requests and not planner and temperature == 0
        if reuse:
            unique, positions, indices = [], {}, []
            for conversation in conversations:
                key = json.dumps(conversation, sort_keys=True, ensure_ascii=False)
                if key not in positions:
                    positions[key] = len(unique)
                    unique.append(conversation)
                indices.append(positions[key])
            self.reused_requests += len(conversations) - len(unique)
            conversations = unique
        # The executor, verifier, generator and tool helpers use the frozen base.
        with nullcontext() if planner else self.model.disable_adapter():
            rows = self.generator(conversations, temperature, max_new_tokens)
        return [dict(rows[i]) for i in indices] if reuse else rows


def action_log_probs(model, samples, temperature, token_chunk_size):
    """No prompt loss; predict every sampled action token, including its first EOS.

    Checkpoint vocabulary projections in small chunks so backpropagation does
    not retain a sequence-length by vocabulary-size float32 tensor.
    """
    device = next(model.parameters()).device
    lengths = [len(s["prompt_ids"]) + len(s["action_ids"]) for s in samples]
    tokens = torch.zeros((len(samples), max(lengths)), dtype=torch.long, device=device)
    mask = torch.zeros_like(tokens)
    for row, (sample, length) in enumerate(zip(samples, lengths)):
        tokens[row, :length] = torch.tensor(sample["prompt_ids"] + sample["action_ids"], device=device)
        mask[row, :length] = 1
    base = model.get_base_model()
    hidden = base.model(input_ids=tokens, attention_mask=mask,
                        use_cache=False, return_dict=True).last_hidden_state
    # Right padding is masked in attention and excluded from every token loss.
    hidden = torch.cat([hidden[row, len(sample["prompt_ids"]) - 1:length - 1]
                        for row, (sample, length) in enumerate(zip(samples, lengths))])
    action_lengths = [len(s["action_ids"]) for s in samples]
    targets = torch.tensor([token for s in samples for token in s["action_ids"]], device=device)

    def project(states, labels):
        logits = base.lm_head(states).float() / temperature
        chosen = logits.gather(-1, labels[:, None]).squeeze(-1)
        return chosen - logits.logsumexp(-1)

    chunks = []
    for offset in range(0, len(targets), token_chunk_size):
        states = hidden[offset:offset + token_chunk_size]
        labels = targets[offset:offset + token_chunk_size]
        if torch.is_grad_enabled():
            chunks.append(checkpoint(project, states, labels, use_reentrant=False))
        else:
            chunks.append(project(states, labels))
    return torch.cat(chunks).split(action_lengths)


def update_planner(model, samples, rewards, settings, progress=print, optimizer=None):
    """Accumulate the complete batch gradient, then call optimizer.step once."""
    from .objective import group_advantages, turn_loss

    advantages = group_advantages(rewards)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer is None:
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    # Save old and frozen-reference likelihoods on CPU before any update.
    model.eval()
    # Preserve the original question/trajectory/turn weights across micro-batches.
    records = [(sample, 1 / (len(samples) * len(group) * len(turns)), advantages[q, r])
               for q, group in enumerate(samples) for r, turns in enumerate(group) for sample in turns]
    total_turns = len(records)
    batch_size = settings.get("update_batch_size", 1)
    for offset in range(0, total_turns, batch_size):
        batch = [sample for sample, _, _ in records[offset:offset + batch_size]]
        with torch.no_grad():
            old = action_log_probs(model, batch, settings["temperature"], settings["logprob_token_chunk"])
            with model.disable_adapter():
                reference = action_log_probs(model, batch, settings["temperature"], settings["logprob_token_chunk"])
            for sample, old_tokens, reference_tokens in zip(batch, old, reference):
                sample["old"] = old_tokens.cpu()
                sample["reference"] = reference_tokens.cpu()
        del old, reference, old_tokens, reference_tokens
        done = offset + len(batch)
        if done % 32 == 0 or done == total_turns:
            progress(f"Prepared likelihoods: {done}/{total_turns} planner turns")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    loss_value = 0.0
    for offset in range(0, total_turns, batch_size):
        batch = records[offset:offset + batch_size]
        current = action_log_probs(model, [sample for sample, _, _ in batch], settings["temperature"],
                                   settings["logprob_token_chunk"])
        loss = sum(weight * turn_loss(tokens, sample.pop("old").to(tokens.device),
                                      sample.pop("reference").to(tokens.device), advantage.to(tokens.device),
                                      settings["clip_epsilon"], settings["kl_beta"])
                   for tokens, (sample, weight, advantage) in zip(current, batch))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Flow-GRPO loss; update stopped.")
        loss.backward()
        loss_value += float(loss.detach())
        del current, loss
        done = offset + len(batch)
        if done % 32 == 0 or done == total_turns:
            progress(f"Accumulated gradients: {done}/{total_turns} planner turns")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
    before = [p.detach().cpu().clone() for p in parameters]
    optimizer.step()
    delta_squared = sum(float((p.detach().cpu() - old).float().square().sum())
                        for p, old in zip(parameters, before))
    optimizer.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    model.eval()
    return dict(optimizer_steps=1, loss=loss_value, gradient_norm=float(grad_norm),
                adapter_change_l2=delta_squared ** 0.5,
                questions_with_reward_variation=int((rewards.std(dim=1) > 0).sum()),
                planner_turns=total_turns,
                planner_action_tokens=sum(len(s["action_ids"]) for g in samples for t in g for s in t),
                trainable_parameters=sum(p.numel() for p in parameters))
