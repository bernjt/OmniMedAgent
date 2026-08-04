import os
import tempfile
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.optim import Optimizer

def save(path: str | Path, model: nn.Module, optimizer: Optimizer, step: int, seed: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "seed": seed}, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def load(path: str | Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
