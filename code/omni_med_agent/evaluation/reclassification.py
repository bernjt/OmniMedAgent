import numpy as np

def nri(target: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> float:
    events = target.astype(bool)
    upward, downward = candidate > reference, candidate < reference
    return float(upward[events].mean() - downward[events].mean() + downward[~events].mean() - upward[~events].mean())

def idi(target: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> float:
    events = target.astype(bool)
    return float(candidate[events].mean() - candidate[~events].mean() - reference[events].mean() + reference[~events].mean())
