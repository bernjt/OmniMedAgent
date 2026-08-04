import torch
from torch import Tensor

def advantages(rewards: Tensor, values: Tensor, dones: Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[Tensor, Tensor]:
    output = torch.zeros_like(rewards)
    running = rewards.new_zeros(rewards.shape[1:])
    following = values[-1]
    for index in range(rewards.shape[0] - 1, -1, -1):
        continuation = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * following * continuation - values[index]
        running = delta + gamma * gae_lambda * continuation * running
        output[index] = running
        following = values[index]
    return output, output + values

def loss(log_probability: Tensor, old_log_probability: Tensor, value: Tensor, returns: Tensor, advantage: Tensor, entropy: Tensor, reference: Tensor, clip: float = 0.2) -> Tensor:
    ratio = (log_probability - old_log_probability).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage).mean()
    return policy + 0.5 * (value - returns).square().mean() - 0.01 * entropy.mean() + 0.02 * (log_probability - reference).mean()
