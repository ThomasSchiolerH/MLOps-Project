from pathlib import Path
import typer
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        print(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)

        # Debug: Print the first few rows of the dataset
        print("Dataset loaded successfully. First 5 rows:")
        print(self.data.head())

    def __len__(self) -> int:
        """Return the length of the dataset."""
        length = len(self.data)
        print("Length of the dataset: ", length)
        return length

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        sample = self.data.iloc[index]
        print(f"Sample at index {index}:")
        print(sample)
        return sample

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Starting preprocessing...")
        
        # Example preprocessing: Drop rows with null values in specific columns
        processed_data = self.data.dropna(subset=["PRICE", "SQM_PRICE"])

        # Debug: Print the first few rows of the processed dataset
        print("Processed dataset (first 5 rows):")
        print(processed_data.head())

        # Save the processed data to the output folder
        output_path = output_folder / self.data_path.name
        processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)

    # Debug: Print dataset length
    print(f"Dataset length: {len(dataset)}")

    # Optional: Retrieve a specific sample for debugging
    _ = dataset.__getitem__(0)  # Retrieve the first sample

    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
