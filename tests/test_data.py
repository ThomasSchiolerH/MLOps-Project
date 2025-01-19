import torch
import pytest
from src.project_name.data import PriceDataset, FEATURE_COLUMNS

# Test data loading
def test_data_loading():
    dataset = PriceDataset(csv_path="../data/processed/test_processed.csv", train=True)
    assert len(dataset) > 0, "Dataset is empty"
    x, y = dataset[0]
    assert x.shape[0] == len(FEATURE_COLUMNS), "Feature count mismatch"
