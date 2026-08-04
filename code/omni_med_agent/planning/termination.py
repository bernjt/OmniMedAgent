from torch import Tensor
from omni_med_agent.types import TaskKind

THRESHOLDS = {TaskKind.DIAGNOSIS: 0.30, TaskKind.RISK: 0.25, TaskKind.REPORT: 0.35, TaskKind.TREATMENT: 0.40, TaskKind.PATHOLOGY: 0.20}

def terminate(entropy: Tensor, tasks: Tensor, step: Tensor, maximum: int = 7) -> Tensor:
    thresholds = entropy.new_tensor([THRESHOLDS[TaskKind(int(task))] for task in tasks])
    return entropy.lt(thresholds) | step.ge(maximum)

def depth_bounds(initial: Tensor, threshold: Tensor, minimum_gain: float = 0.12, maximum_gain: float = 0.82) -> tuple[Tensor, Tensor]:
    remaining = (initial - threshold).clamp_min(0.0)
    return remaining / maximum_gain, remaining / minimum_gain
