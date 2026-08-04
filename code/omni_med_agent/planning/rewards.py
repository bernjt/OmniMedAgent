from torch import Tensor

def information_gain(previous: Tensor, current: Tensor) -> Tensor:
    return previous - current

def diagnostic_reward(prediction: Tensor, target: Tensor) -> Tensor:
    return prediction.eq(target).float()

def risk_reward(probability: Tensor, target: Tensor) -> Tensor:
    return 1.0 - (probability - target.float()).abs()

def composite(information: Tensor, task: Tensor, cost: Tensor, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.01) -> Tensor:
    return alpha * information + beta * task - gamma * cost
