import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project_name.data import PriceDataset, FEATURE_COLUMNS
from project_name.model import PricePredictionModel

def train_model(
    train_csv: str = "../../data/processed/train_processed.csv",
    val_csv: str = "../../data/processed/val_processed.csv",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    random_seed: int = 42
):
    """
    Train a simple feed-forward neural network with separate train and val CSVs.
    Assumes data.py has already split the data and scaled the features.
    """
    torch.manual_seed(random_seed)

    # 1. Load the separate processed datasets
    train_dataset = PriceDataset(csv_path=train_csv, train=True)
    val_dataset   = PriceDataset(csv_path=val_csv,   train=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 2. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # 3. Initialize model + optimizer
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)  # [batch_size, 1]
            loss = criterion(preds.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds.squeeze(), batch_y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Train Loss: {avg_train_loss:.4f} "
              f"- Val Loss: {avg_val_loss:.4f}")

    # 5. Final validation MSE
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y)
            val_loss += loss.item()
    final_val_loss = val_loss / len(val_loader)
    print(f"Final validation MSE after {epochs} epochs: {final_val_loss:.4f}")

    return model
