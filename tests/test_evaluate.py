import torch
import pytest
from src.AVM.model import PricePredictionModel

# Test accuracy metric calculation
def test_accuracy_calculation():
    preds = torch.tensor([100.0, 110.0, 95.0])
    targets = torch.tensor([100.0, 120.0, 90.0])

    def calculate_accuracy_margin(preds, targets, margin):
        within_margin = torch.abs(preds - targets) <= (targets * margin)
        return within_margin.float().mean().item() * 100

    # Test for ±10% margin
    acc_10 = calculate_accuracy_margin(preds, targets, 0.10)
    assert acc_10 == pytest.approx(100.0), "Accuracy calculation for ±10% margin failed"

    # Test for ±5% margin
    acc_5 = calculate_accuracy_margin(preds, targets, 0.05)
    assert acc_5 == pytest.approx(33.33333333333333), "Accuracy calculation for ±5% margin failed"

    # Test for ±20% margin
    acc_20 = calculate_accuracy_margin(preds, targets, 0.20)
    assert acc_20 == pytest.approx(100.0), "Accuracy calculation for ±20% margin failed"

# Test model evaluation logic with mocked components
def test_model_evaluation_logic():
    # Create a dummy model
    model = PricePredictionModel(in_features=24)
    model.eval()

    # Mock dummy data
    dummy_inputs = torch.randn(10, 24)  # 10 samples, 24 features each
    dummy_targets = torch.randn(10)  # 10 target values

    # Define a dummy criterion
    criterion = torch.nn.MSELoss()

    # Perform a forward pass and calculate loss
    with torch.no_grad():
        preds = model(dummy_inputs).squeeze()
        loss = criterion(preds, dummy_targets)

    # Assert loss is a non-negative scalar
    assert loss.item() >= 0, "Loss should be non-negative"

    # Optionally test predicted output shape
    assert preds.shape == dummy_targets.shape, "Prediction and target shapes should match"

# Test model evaluation with edge cases
def test_model_edge_cases():
    # Create a dummy model
    model = PricePredictionModel(in_features=24)
    model.eval()

    # Edge case: Empty input tensor
    empty_input = torch.empty(0, 24)  # No samples, 24 features
    with torch.no_grad():
        empty_output = model(empty_input)
        assert empty_output.shape == (0, 1), "Output shape should match empty input"

    # Edge case: Single input sample
    single_input = torch.randn(1, 24)  # Single sample
    with torch.no_grad():
        single_output = model(single_input)
        assert single_output.shape == (1, 1), "Output shape for single input sample is incorrect"
