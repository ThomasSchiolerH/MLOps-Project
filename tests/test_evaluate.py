import torch
import pytest
from src.project_name.model import PricePredictionModel
from src.project_name.data import PriceDataset, FEATURE_COLUMNS
from project_name.evaluate import evaluate_model

def test_evaluate_model():
    mse = evaluate_model(model_checkpoint="model.pth", csv_path="test_data.csv")
    assert mse >= 0, "MSE should be non-negative"
    
    
# Test accuracy metric calculation
def test_accuracy_calculation():
    preds = torch.tensor([100, 110, 95])
    targets = torch.tensor([100, 120, 90])

    def calculate_accuracy_margin(preds, targets, margin):
        within_margin = torch.abs(preds - targets) <= (targets * margin)
        return within_margin.float().mean().item() * 100

    acc_10 = calculate_accuracy_margin(preds, targets, 0.10)
    assert acc_10 == 66.66666666666666, "Accuracy calculation failed"    