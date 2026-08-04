import torch
from torch import Tensor, nn
from omni_med_agent.types import EncodedEvidence, SparseMass

class MassAssignment(nn.Module):
    def __init__(self, hidden_size: int, focal_count: int, hypothesis_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, focal_count))
        masks = [1 << index for index in range(hypothesis_count)]
        masks += [(1 << left) | (1 << right) for left in range(hypothesis_count) for right in range(left + 1, hypothesis_count)]
        self.register_buffer("masks", torch.tensor(masks[:focal_count], dtype=torch.long))

    def forward(self, evidence: EncodedEvidence) -> SparseMass:
        confidence = evidence.confidence.clamp(0.0, 1.0)
        values = self.network(evidence.representation).softmax(-1) * confidence.unsqueeze(-1)
        return SparseMass(self.masks.unsqueeze(0).expand(values.shape[0], -1), values, 1.0 - confidence)

def normalization_error(mass: SparseMass) -> Tensor:
    return (mass.values.sum(-1) + mass.universal - 1.0).square().mean()
