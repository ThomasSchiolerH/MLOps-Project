import torch
from src.AVM.evaluate import evaluate_model

def test_evaluate_model():
    mse, acc_5, acc_10, acc_20 = evaluate_model(
        model_checkpoint="models/price_model.pth", 
        csv_path="data/processed/test_processed.csv")
    assert mse >= 0, "MSE should be non-negative"
    
# Test accuracy metric calculation
def test_accuracy_calculation():
    preds = torch.tensor([100, 110, 95])
    targets = torch.tensor([100, 120, 90])

    def calculate_accuracy_margin(preds, targets, margin):
        within_margin = torch.abs(preds - targets) <= (targets * margin)
        return within_margin.float().mean().item() * 100

    acc_10 = calculate_accuracy_margin(preds, targets, 0.10)
    assert acc_10 == 100.0, "Accuracy calculation failed"    