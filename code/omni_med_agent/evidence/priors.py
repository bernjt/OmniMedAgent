import torch
from torch import Tensor
from omni_med_agent.types import SparseMass, TaskKind

STRENGTH = {TaskKind.DIAGNOSIS: 0.6, TaskKind.RISK: 0.4, TaskKind.REPORT: 0.0, TaskKind.TREATMENT: 0.6, TaskKind.PATHOLOGY: 0.0}

def initialize(probabilities: Tensor, tasks: Tensor) -> SparseMass:
    hypotheses = probabilities.shape[-1]
    strengths = probabilities.new_tensor([STRENGTH[TaskKind(int(task))] for task in tasks])
    values = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8) * strengths.unsqueeze(-1)
    masks = torch.tensor([1 << index for index in range(hypotheses)], device=probabilities.device).unsqueeze(0).expand(probabilities.shape[0], -1)
    return SparseMass(masks, values, 1.0 - strengths)
