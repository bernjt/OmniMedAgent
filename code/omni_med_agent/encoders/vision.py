import torch
from torch import Tensor, nn
from omni_med_agent.types import EncodedEvidence, ModalityKind

class VisionEncoder(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(3, width, 16, 16)
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.confidence = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, images: Tensor) -> EncodedEvidence:
        tokens = self.input(images).flatten(2).transpose(1, 2)
        representation = self.norm(self.encoder(tokens).mean(1))
        confidence = self.confidence(representation).squeeze(-1).sigmoid()
        modality = torch.full((representation.shape[0],), int(ModalityKind.IMAGING), device=representation.device)
        return EncodedEvidence(representation, confidence, modality)
