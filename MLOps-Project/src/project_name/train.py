import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from project_name.data import PriceDataset, FEATURE_COLUMNS
from project_name.model import PricePredictionModel

def train_model(
    csv_path: str = "data/processed/HACKATHON.AVM_EJERLEJLIGHEDER_TRAIN.csv",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_split_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    Train a simple feed-forward neural network, 
    using an 80/20 train/validation split.
    """
    torch.manual_seed(random_seed)

    # 1. Create the full dataset (labeled data)
    dataset = PriceDataset(csv_path=csv_path)

    # 2. Split dataset into train/val subsets
    dataset_size = len(dataset)
    train_size = int(train_split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 3. Make DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # 4. Initialize model + optimizer
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training loop
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

        # Validation for this epoch
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

    # 6. Final check on the validation set (optional, for clarity)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y)
            val_loss += loss.item()
    final_val_loss = val_loss / len(val_loader)
    print(f"Final validation MSE after {epochs} epochs: {final_val_loss:.4f}")

    # Return the trained model so we can save it
    return model
