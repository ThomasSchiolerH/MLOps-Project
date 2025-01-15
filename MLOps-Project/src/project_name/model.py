import torch
from torch import nn


class PricePredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc1(x)


if __name__ == "__main__":
    model = PricePredictionModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

