from collections import defaultdict
import torch
from torch import Tensor
from omni_med_agent.types import SparseMass

def _one(lm: Tensor, lv: Tensor, lu: Tensor, rm: Tensor, rv: Tensor, ru: Tensor, hypotheses: int, threshold: float, limit: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    universe = (1 << hypotheses) - 1
    left = list(zip(lm.tolist(), lv.tolist())) + [(universe, float(lu))]
    right = list(zip(rm.tolist(), rv.tolist())) + [(universe, float(ru))]
    products: dict[int, float] = defaultdict(float)
    conflict = 0.0
    for a, x in left:
        for b, y in right:
            intersection = a & b
            if intersection:
                products[intersection] += x * y
            else:
                conflict += x * y
    if conflict > threshold:
        return lm, lv, lu, lv.new_tensor(conflict)
    scale = max(1.0 - conflict, 1e-8)
    universal = products.pop(universe, 0.0) / scale
    ranked = sorted(products.items(), key=lambda item: item[1], reverse=True)[:limit]
    masks = lm.new_tensor([item[0] for item in ranked])
    values = lv.new_tensor([item[1] / scale for item in ranked])
    universal += max(0.0, 1.0 - float(values.sum()) - universal)
    return masks, values, lu.new_tensor(universal), lv.new_tensor(conflict)

def combine(left: SparseMass, right: SparseMass, hypotheses: int, threshold: float = 0.3, limit: int = 100) -> tuple[SparseMass, Tensor]:
    items = [_one(left.masks[i], left.values[i], left.universal[i], right.masks[i], right.values[i], right.universal[i], hypotheses, threshold, limit) for i in range(left.values.shape[0])]
    width = max(item[0].numel() for item in items)
    masks = left.masks.new_zeros((len(items), width))
    values = left.values.new_zeros((len(items), width))
    universal = left.universal.new_zeros(len(items))
    for i, item in enumerate(items):
        masks[i, :item[0].numel()] = item[0]
        values[i, :item[1].numel()] = item[1]
        universal[i] = item[2]
    return SparseMass(masks, values, universal), torch.stack([item[3] for item in items])
