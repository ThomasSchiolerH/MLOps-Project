import torch
from torch import nn

class PricePredictionModel(nn.Module):
    """
    Feed-forward neural network for SQM price prediction.
    """

    def __init__(self, in_features: int = 24, hidden1: int = 64, hidden2: int = 32, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, in_features]
        :return: [batch_size, 1] (predicted SQM price)
        """
        return self.net(x)


if __name__ == "__main__":
    # Simple test
    model = PricePredictionModel(in_features=24)
    print(model)

    dummy_input = torch.randn(2, 24)  # batch=2, 24 features
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (2, 1)
