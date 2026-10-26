import torch
import torch.nn.functional as F


def group_advantages(rewards, group_ids, floor=0.05):
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must both have shape [responses]")
    if not torch.isfinite(rewards).all() or floor <= 0:
        raise ValueError("Nonfinite rewards or invalid standard deviation floor")
    result = torch.zeros_like(rewards)
    for group in group_ids.unique():
        mask = group_ids == group
        values = rewards[mask]
        if values.numel() < 2:
            raise ValueError("Incomplete rollout group")
        result[mask] = (values - values.mean()) / values.std(unbiased=False).clamp_min(floor)
    return result.detach()


def shape_advantages(advantages, penalty, weight, kind="additive"):
    if penalty.ndim != 2 or advantages.shape != (penalty.shape[0],):
        raise ValueError("Expected advantages [N], penalties [N,T]")
    if not torch.isfinite(penalty).all() or (penalty < 0).any() or (penalty > 1).any():
        raise ValueError("Penalty must be finite in [0,1]")
    if weight < 0:
        raise ValueError("Penalty weight cannot be negative")
    a = advantages[:, None]
    if kind == "additive":
        return (a - weight * penalty).detach()
    if kind == "multiplicative" and weight <= 1:
        return (a - weight * a.abs() * penalty).detach()
    raise ValueError("Invalid penalty type or multiplicative weight")


def clipped_surrogate(log_probs, old_log_probs, advantages, epsilon):
    if log_probs.shape != old_log_probs.shape or advantages.shape != log_probs.shape:
        raise ValueError("Token tensors must have identical shapes")
    ratio = (log_probs - old_log_probs.detach()).exp()
    loss = -torch.minimum(ratio * advantages,
                          ratio.clamp(1-epsilon, 1+epsilon) * advantages)
    return loss, ratio


def forward_kl(logits, reference_logits):
    """Exact categorical KL at given prefixes, with gradients only through policy."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    log_ref = F.log_softmax(reference_logits.detach().float(), dim=-1)
    return (log_p.exp() * (log_p - log_ref)).sum(-1)


def token_loss(logits, reference_logits, actions, old_log_probs, advantages, epsilon, beta):
    log_p = F.log_softmax(logits.float(), dim=-1)
    chosen = log_p.gather(-1, actions[:, None]).squeeze(-1)
    pg, ratio = clipped_surrogate(chosen, old_log_probs, advantages, epsilon)
    kl = forward_kl(logits, reference_logits) if beta else torch.zeros_like(pg)
    return pg + beta * kl, chosen, kl, ratio
