import torch
from torch import nn

class PricePredictionModel(nn.Module):
    """
    Feed-forward neural network for SQM price prediction with more layers.
    """

    def __init__(
            self,
            in_features: int = 22,
            hidden1: int = 128,
            hidden2: int = 64,
            hidden3: int = 32,
            dropout_p: float = 0.2
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden1),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),

                nn.Linear(hidden1, hidden2),
                #add sigmoid activation function
                nn.Sigmoid(),
                nn.Dropout(p=dropout_p),

                nn.Linear(hidden2, hidden3),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),

                nn.Linear(hidden3, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



if __name__ == "__main__":
    # Simple test
    model = PricePredictionModel(in_features=22)
    print(model)

    dummy_input = torch.randn(2, 22)  # batch=2, 24 features
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (2, 1)
