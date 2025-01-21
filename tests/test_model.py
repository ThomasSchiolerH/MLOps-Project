import torch
from src.AVM.model import PricePredictionModel

def test_model_initialization():
    # Test if the model initializes correctly
    model = PricePredictionModel(in_features=24)
    assert isinstance(model, PricePredictionModel), "Model failed to initialize properly"

def test_model_forward_pass():
    # Test the forward pass with dummy input
    in_features = 24
    batch_size = 2
    model = PricePredictionModel(in_features=in_features)

    # Create dummy input tensor with batch_size rows and in_features columns
    dummy_input = torch.randn(batch_size, in_features)

    # Perform forward pass
    output = model(dummy_input)

    # Assert output shape is as expected
    assert output.shape == (batch_size, 1), f"Output shape is incorrect: {output.shape}"

def test_dropout_layers():
    # Test if dropout layers are applied
    in_features = 24
    model = PricePredictionModel(in_features=in_features, dropout_p=0.5)

    # Ensure dropout layers exist in the model's structure
    dropout_layers = [layer for layer in model.net if isinstance(layer, torch.nn.Dropout)]
    assert len(dropout_layers) > 0, "Model should have at least one Dropout layer"

def test_relu_activation():
    # Test if ReLU activations are present
    in_features = 24
    model = PricePredictionModel(in_features=in_features)

    # Ensure ReLU layers exist in the model's structure
    relu_layers = [layer for layer in model.net if isinstance(layer, torch.nn.ReLU)]
    assert len(relu_layers) > 0, "Model should have at least one ReLU activation layer"
