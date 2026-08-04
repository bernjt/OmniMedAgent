import torch
from torch import Tensor
from omni_med_agent.types import BeliefState, SparseMass

def intervals(mass: SparseMass, hypotheses: int) -> tuple[Tensor, Tensor]:
    belief = []
    plausibility = []
    for index in range(hypotheses):
        bit = 1 << index
        belief.append((mass.values * mass.masks.eq(bit)).sum(-1))
        plausibility.append((mass.values * torch.bitwise_and(mass.masks, bit).ne(0)).sum(-1) + mass.universal)
    return torch.stack(belief, -1), torch.stack(plausibility, -1)

def pignistic(mass: SparseMass, hypotheses: int) -> Tensor:
    output = mass.values.new_zeros((mass.values.shape[0], hypotheses))
    cardinality = mass.masks.clone()
    counts = cardinality.new_zeros(cardinality.shape)
    for index in range(hypotheses):
        counts += torch.bitwise_and(cardinality, 1 << index).ne(0)
    for index in range(hypotheses):
        output[:, index] = (mass.values * torch.bitwise_and(mass.masks, 1 << index).ne(0) / counts.clamp_min(1)).sum(-1) + mass.universal / hypotheses
    return output / output.sum(-1, keepdim=True).clamp_min(1e-8)

def entropy(mass: SparseMass, hypotheses: int) -> Tensor:
    probability = pignistic(mass, hypotheses)
    return -(probability * probability.clamp_min(1e-8).log2()).sum(-1)

def state(mass: SparseMass, conflict: Tensor, hypotheses: int) -> BeliefState:
    belief, plausibility = intervals(mass, hypotheses)
    return BeliefState(belief, plausibility, conflict, entropy(mass, hypotheses), mass)
