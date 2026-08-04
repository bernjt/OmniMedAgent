import torch
from torch import Tensor, nn
from omni_med_agent.types import BeliefState, PolicyOutput

class HierarchicalPolicy(nn.Module):
    def __init__(self, hypotheses: int, context: int, hidden: int, strategies: int = 7, actions: int = 23) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hypotheses * 2 + context + 2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.strategy = nn.Linear(hidden, strategies)
        self.strategy_embedding = nn.Embedding(strategies, hidden)
        self.sequence = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, actions))
        self.value = nn.Linear(hidden, 1)

    def forward(self, state: BeliefState, context: Tensor, step: Tensor, available: Tensor, strategy: Tensor | None = None) -> PolicyOutput:
        encoded = self.encoder(torch.cat((state.belief, state.plausibility, context, state.conflict.unsqueeze(-1), step.float().unsqueeze(-1) / 7.0), -1))
        strategy_logits = self.strategy(encoded)
        if strategy is None:
            strategy = torch.distributions.Categorical(logits=strategy_logits).sample()
        action_logits = self.sequence(torch.cat((encoded, self.strategy_embedding(strategy)), -1)).masked_fill(~available, torch.finfo(encoded.dtype).min)
        return PolicyOutput(strategy_logits, action_logits, self.value(encoded).squeeze(-1))
