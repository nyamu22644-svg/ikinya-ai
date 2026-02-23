import torch

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)