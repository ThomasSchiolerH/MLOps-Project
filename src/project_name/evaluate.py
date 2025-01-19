import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project_name.data import PriceDataset, FEATURE_COLUMNS
from project_name.model import PricePredictionModel

def evaluate_model(model_checkpoint: str, csv_path: str) -> float:
    """
    Load a model checkpoint and evaluate it on a given dataset.
    Returns the mean-squared error.
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

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)  # [batch_size, 1]
            loss = criterion(preds.squeeze(), batch_y)
            total_loss += loss.item()

    mse = total_loss / len(data_loader)
    print(f"Mean Squared Error on {csv_path}: {mse:.4f}")
    return mse
