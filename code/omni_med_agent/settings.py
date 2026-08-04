from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class ModelSettings:
    hidden_size: int = 1024
    hypothesis_count: int = 50
    focal_count: int = 100
    strategy_count: int = 7
    action_count: int = 23
    conflict_threshold: float = 0.3

@dataclass(frozen=True)
class PlannerSettings:
    max_steps: int = 7
    information_gain_weight: float = 0.5
    task_weight: float = 0.4
    efficiency_weight: float = 0.01
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    kl_weight: float = 0.02

@dataclass(frozen=True)
class TrainSettings:
    stage: str = "rl"
    batch_size: int = 16
    world_size: int = 8
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 10000
    precision: str = "bf16"
    seed: int = 3407

@dataclass(frozen=True)
class DataSettings:
    manifest: str = "data/manifest.csv"
    image_size: int = 448
    tile_size: int = 256
    text_length: int = 2048

@dataclass(frozen=True)
class ExperimentSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    data: DataSettings = field(default_factory=DataSettings)

def _make(kind: type[Any], values: dict[str, Any]) -> Any:
    return kind(**{key: value for key, value in values.items() if key in kind.__dataclass_fields__})

def load_settings(path: str | Path) -> ExperimentSettings:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return ExperimentSettings(_make(ModelSettings, values.get("model", {})), _make(PlannerSettings, values.get("planner", {})), _make(TrainSettings, values.get("train", {})), _make(DataSettings, values.get("data", {})))
