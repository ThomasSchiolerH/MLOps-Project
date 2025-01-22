import typer
import torch
from pathlib import Path
from google.cloud import storage

from data import preprocess
from train import train_model
from evaluate import evaluate_model

app = typer.Typer()

@app.command()
def data(
    raw_data_path: str = "data/raw/HACKATHON.AVM_EJERLEJLIGHEDER_TRAIN.csv",
    output_folder: str = "data/processed"
):
    """
    Run a simple data preprocessing step to output a clean CSV.
    """
    preprocess(Path(raw_data_path), Path(output_folder))

@app.command()
def train(
    train_csv: str = "data/processed/train_processed.csv",
    val_csv: str = "data/processed/val_processed.csv",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_output: str = "src/AVM/price_model.pth",
    random_seed: int = 42
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
        lr=lr,
        random_seed=random_seed
    )
    torch.save(model.state_dict(), model_output)
    

    def upload_to_gcs(local_path, bucket_name, destination_blob_name):
        """Uploads a file to Google Cloud Storage."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        print(f"Model uploaded to gs://{bucket_name}/{destination_blob_name}")

    
    upload_to_gcs("src/AVM/price_model.pth", "avm-storage", "models/price_model_vertexworking3.pth")

    print(f"Model weights saved to {model_output}")

@app.command()
def evaluate(
    
    model_checkpoint: str = "src/AVM/price_model.pth",
    test_file: str = "data/processed/test_processed.csv"
):
    """
    Evaluate the trained model on a test dataset (MSE).
    """
    client = storage.Client()
    bucket = client.bucket("avm-storage")
    blob = bucket.blob("models/price_model_vertexworking3.pth")

    # Download the model locally
    blob.download_to_filename("src/AVM/vertex.pth")
    print("Model downloaded successfully.")

    evaluate_model("src/AVM/vertex.pth", test_file)

if __name__ == "__main__":
    app()
