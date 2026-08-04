import numpy as np
from scipy import stats

def bootstrap(values: np.ndarray, resamples: int = 10000, seed: int = 3407) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.array([generator.choice(values, values.size, replace=True).mean() for _ in range(resamples)])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)

def permutation(left: np.ndarray, right: np.ndarray, permutations: int = 10000, seed: int = 3407) -> float:
    differences = left - right
    generator = np.random.default_rng(seed)
    observed = abs(differences.mean())
    exceed = sum(abs((differences * generator.choice([-1.0, 1.0], differences.size)).mean()) >= observed for _ in range(permutations))
    return (exceed + 1.0) / (permutations + 1.0)

def wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    return float(stats.wilcoxon(left, right).pvalue)

def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    return float(difference.mean() / difference.std(ddof=1))
