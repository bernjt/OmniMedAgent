import torch
from torch import Tensor, nn
from omni_med_agent.types import EncodedEvidence, ModalityKind

class TextEncoder(nn.Module):
    def __init__(self, input_size: int, width: int, depth: int) -> None:
        super().__init__()
        self.input = nn.Embedding(input_size, width)
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.confidence = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, values: Tensor) -> EncodedEvidence:
        tokens = self.input(values)
        representation = self.norm(self.encoder(tokens).mean(1))
        confidence = self.confidence(representation).squeeze(-1).sigmoid()
        modality = torch.full((representation.shape[0],), int(ModalityKind.NOTES), device=representation.device)
        return EncodedEvidence(representation, confidence, modality)
