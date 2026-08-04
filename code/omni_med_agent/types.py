from dataclasses import dataclass
from enum import IntEnum
from torch import Tensor

class TaskKind(IntEnum):
    DIAGNOSIS = 0
    RISK = 1
    REPORT = 2
    TREATMENT = 3
    PATHOLOGY = 4

class ModalityKind(IntEnum):
    IMAGING = 0
    LABORATORY = 1
    NOTES = 2
    PATHOLOGY = 3
    VITALS = 4

@dataclass
class EncodedEvidence:
    representation: Tensor
    confidence: Tensor
    modality: Tensor

@dataclass
class SparseMass:
    masks: Tensor
    values: Tensor
    universal: Tensor

@dataclass
class BeliefState:
    belief: Tensor
    plausibility: Tensor
    conflict: Tensor
    entropy: Tensor
    mass: SparseMass

@dataclass
class PolicyOutput:
    strategy_logits: Tensor
    action_logits: Tensor
    value: Tensor
