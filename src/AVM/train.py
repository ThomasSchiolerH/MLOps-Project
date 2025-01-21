import yaml
import torch
import wandb
from data import PriceDataset, FEATURE_COLUMNS
from model import PricePredictionModel

# Default configuration file path
DEFAULT_CONFIG_PATH = "../../configs/modelTraningConfig.yaml"

def load_config(config_path: str, config_name: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config["train_configs"][config_name]

def train_model(train_csv, val_csv, epochs, batch_size, lr, random_seed, log_to_wandb=True):
    torch.manual_seed(random_seed)

    # Initialize W&B
    if log_to_wandb:
        wandb.init(project="sqm-price-prediction", config={
            "train_csv": train_csv,
            "val_csv": val_csv,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "random_seed": random_seed
        })

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

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Log metrics to W&B
        if log_to_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

    if log_to_wandb:
        wandb.finish()

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Name of the configuration to use (e.g., 'baseline', 'large_batch', or 'high_lr')"
    )
    args = parser.parse_args()

    # Load specific configuration from the default YAML file
    config = load_config(DEFAULT_CONFIG_PATH, args.config)

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
