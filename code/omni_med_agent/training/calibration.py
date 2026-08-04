import torch
from torch import Tensor

def expected_error(confidence: Tensor, correctness: Tensor, bins: int = 15) -> Tensor:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=confidence.device)
    result = confidence.new_zeros(())
    for index in range(bins):
        selected = confidence.ge(edges[index]) & confidence.lt(edges[index + 1])
        if selected.any():
            result += selected.float().mean() * (confidence[selected].mean() - correctness[selected].float().mean()).abs()
    return result
