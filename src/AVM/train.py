import yaml
import argparse
import torch
from data import PriceDataset, FEATURE_COLUMNS
from model import PricePredictionModel

def load_config(config_path: str, config_name: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config["train_configs"][config_name]

def train_model(train_csv, val_csv, epochs, batch_size, lr, random_seed):
    torch.manual_seed(random_seed)

    # Load datasets
    train_dataset = PriceDataset(csv_path=train_csv, train=True)
    val_dataset = PriceDataset(csv_path=val_csv, train=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds.squeeze(), batch_y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss/len(val_loader):.4f}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="../../configs/modelTrainingConfig.yaml", help="Path to config file")
    parser.add_argument('--config_setting', type=str, required=True, help="Name of the configuration to use")
    args = parser.parse_args()

    # Load specific configuration
    config = load_config(args.config, args.config_name)

    # Train model with the selected configuration
    model = train_model(
        train_csv=config["train_csv"],
        val_csv=config["val_csv"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        random_seed=config["random_seed"]
    )

    # Save the model
    torch.save(model.state_dict(), config["model_output"])
    print(f"Model saved to {config['model_output']}")
