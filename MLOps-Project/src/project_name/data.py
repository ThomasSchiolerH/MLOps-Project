from pathlib import Path
import typer
import pandas as pd
import torch
from torch.utils.data import Dataset

# Example: columns used for features
FEATURE_COLUMNS = [
    "AREA_TINGLYST",
    "AREA_RESIDENTIAL",
    "NUMBER_ROOMS",
    "DISTANCE_LAKE",
    "DISTANCE_HARBOUR",
    "DISTANCE_COAST",
    "CONSTRUCTION_YEAR",
]
TARGET_COLUMN = "SQM_PRICE"


class PriceDataset(Dataset):
    """
    A custom PyTorch Dataset for apartment price data.
    Returns (features, target) as torch.Tensors.
    """

    def __init__(self, csv_path: Path, train: bool = True) -> None:
        """
        :param csv_path: Path to the CSV file.
        :param train: Flag indicating if this is a training or test dataset.
                      You might handle different logic or transforms based on this.
        """
        self.csv_path = csv_path
        print(f"Loading data from: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)

        # Example minimal preprocessing
        # Drop rows missing the target, fill other NAs with 0
        self.data = self.data.dropna(subset=[TARGET_COLUMN]).fillna(0)

        # Basic debugging info
        print("Dataset loaded. First 5 rows:")
        print(self.data.head())

        # Convert feature columns to float
        self.features = self.data[FEATURE_COLUMNS].values.astype("float32")

        # Convert target column to float
        self.targets = self.data[TARGET_COLUMN].values.astype("float32")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Return (features, target) as torch Tensors.
        """
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """
    A placeholder function illustrating a data preprocessing step.
    This could be more elaborate (imputations, data cleaning, etc.).
    """
    print("Starting data preprocessing...")
    df = pd.read_csv(raw_data_path)

    # Example: Drop rows with null in crucial columns
    df = df.dropna(subset=["PRICE", "SQM_PRICE"])
    # Optionally fill other columns
    df = df.fillna(0)

    output_path = output_folder / raw_data_path.name
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to: {output_path}")


if __name__ == "__main__":
    typer.run(preprocess)
