from torch import Tensor, nn

class PathologyHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, outputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size, outputs))

    def forward(self, values: Tensor) -> Tensor:
        logits = self.network(values)
        return logits
