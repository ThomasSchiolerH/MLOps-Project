import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.AVM.data import PriceDataset, FEATURE_COLUMNS
from src.AVM.model import PricePredictionModel

def evaluate_model(model_checkpoint: str, csv_path: str) -> float:
    """
    Load a model checkpoint and evaluate it on a given dataset.
    Returns the mean-squared error and additional accuracy metrics.
    """
    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Create dataset and DataLoader
    dataset = PriceDataset(csv_path=csv_path, train=True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    mse = total_loss / len(data_loader)
    print(f"Mean Squared Error on {csv_path}: {mse:.4f}")

    # Calculate additional accuracy metrics
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)

    def calculate_accuracy_margin(preds, targets, margin):
        within_margin = torch.abs(preds - targets) <= (targets * margin)
        return within_margin.float().mean().item() * 100  # Convert to percentage

    acc_5 = calculate_accuracy_margin(all_preds, all_targets, 0.05)
    acc_10 = calculate_accuracy_margin(all_preds, all_targets, 0.10)
    acc_20 = calculate_accuracy_margin(all_preds, all_targets, 0.20)

    print(f"Accuracy within ±5%: {acc_5:.2f}% (Benchmark: 30.1%)")
    print(f"Accuracy within ±10%: {acc_10:.2f}% (Benchmark: 54.1%)")
    print(f"Accuracy within ±20%: {acc_20:.2f}% (Benchmark: 81.4%)")

    return mse, acc_5, acc_10, acc_20
