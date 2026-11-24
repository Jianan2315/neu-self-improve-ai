"""Flow-GRPO token loss; the trainer averages tokens, turns, rollouts, questions."""
import torch


def group_advantages(rewards):
    """Rows are questions; columns are their sampled trajectories."""
    return ((rewards - rewards.mean(dim=1, keepdim=True)) /
            (rewards.std(dim=1, correction=1, keepdim=True) + 1e-6)).detach()


def turn_loss(current, old, reference, advantage, clip_epsilon, beta):
    ratio = (current - old.detach()).exp()
    policy = torch.minimum(ratio * advantage,
                           ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    difference = reference.detach() - current
    sampled_kl = difference.exp() - difference - 1
    return (-policy + beta * sampled_kl).mean()
