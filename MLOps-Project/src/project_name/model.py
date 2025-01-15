import torch
from torch import nn

class PricePredictionModel(nn.Module):
    """
    A simple feed-forward neural network for price prediction.
    """

    def __init__(self, in_features: int = 7):
        super().__init__()
        # Example: 1 hidden layer of size 32
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :param x: [batch_size, in_features] float tensor
        :return: [batch_size, 1] float tensor
        """
        return self.net(x)


if __name__ == "__main__":
    # Quick test of the model
    model = PricePredictionModel(in_features=7)
    print(model)

    # Create a dummy input (batch of 2, 7 features)
    dummy_input = torch.randn(2, 7)
    output = model(dummy_input)
    print("Output shape:", output.shape)
