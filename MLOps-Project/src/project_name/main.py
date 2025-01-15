import typer
import torch
from pathlib import Path

from project_name.data import preprocess
from project_name.train import train_model
from project_name.evaluate import evaluate_model

app = typer.Typer()

@app.command()
def data_preprocess(
    raw_data_path: str = "../data/raw/HACKATHON.AVM_EJERLEJLIGHEDER_TRAIN.csv",
    output_folder: str = "../data/processed"
):
    """
    Run a simple data preprocessing step to output a clean CSV.
    """
    preprocess(Path(raw_data_path), Path(output_folder))

@app.command()
def train(
    train_csv: str = "../data/processed/train_processed.csv",
    val_csv: str = "../data/processed/val_processed.csv",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_output: str = "../models/price_model.pth"
):
    """
    Train a simple PyTorch feed-forward model using 
    the preprocessed train and validation CSVs.
    """
    model = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    torch.save(model.state_dict(), model_output)
    print(f"Model weights saved to {model_output}")

@app.command()
def evaluate(
    model_checkpoint: str = "models/price_model.pth",
    test_file: str = "data/raw/HACKATHON.AVM_EJERLEJLIGHEDER_TEST.csv"
):
    """
    Evaluate the trained model on a test dataset (MSE).
    """
    mse = evaluate_model(model_checkpoint, test_file)
    print(f"Final test MSE: {mse:.4f}")

if __name__ == "__main__":
    app()
