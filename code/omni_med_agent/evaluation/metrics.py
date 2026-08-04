import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from torch import Tensor

def mean_auroc(target: Tensor, probability: Tensor) -> float:
    y, p = target.cpu().numpy(), probability.cpu().numpy()
    values = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if np.unique(y[:, i]).size > 1]
    return float(np.mean(values))

def accuracy(target: Tensor, logits: Tensor) -> float:
    return float(logits.argmax(-1).eq(target).float().mean())

def entity_f1(reference: set[str], generated: set[str]) -> float:
    vocabulary = sorted(reference | generated)
    if not vocabulary:
        return 1.0
    return float(f1_score([int(x in reference) for x in vocabulary], [int(x in generated) for x in vocabulary], zero_division=0))

def concordance(first_line: Tensor, alternative: Tensor) -> float:
    return float((0.6 * first_line.float() + 0.4 * alternative.float()).mean())
