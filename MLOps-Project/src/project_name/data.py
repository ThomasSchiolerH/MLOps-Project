from pathlib import Path
import typer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Column definitions
# -----------------------------
AREA_COLUMNS = [
    "AREA_TINGLYST",
    "AREA_RESIDENTIAL",
    "AREA_OTHER",
    "AREA_COMMON_ACCESS_SHARE",
    "AREA_CLOSED_COVER_OUTHOUSE",
    "AREA_OPEN_BALCONY_ROOFTOP",
]
FACILITY_COLUMNS = [
    "FACILITIES_TOILET",
    "FACILITIES_SHOWER",
    "FACILITIES_KITCHEN",
]
DROP_COLUMNS = [
    "TRANSACTION_ID",
    "UNIT_ID",
    "BUILDING_ID",
    "STREET_CODE",
]
FEATURE_COLUMNS = [
    "FLOOR",
    "CONSTRUCTION_YEAR",
    "REBUILDING_YEAR",
    "DISTANCE_LAKE",
    "DISTANCE_HARBOUR",
    "DISTANCE_COAST",
    "HAS_ELEVATOR",
    "HAS_TOILET",
    "HAS_SHOWER",
    "HAS_KITCHEN",
    "AREA_TINGLYST",
    "AREA_RESIDENTIAL",
    "AREA_OTHER",
    "AREA_COMMON_ACCESS_SHARE",
    "AREA_CLOSED_COVER_OUTHOUSE",
    "AREA_OPEN_BALCONY_ROOFTOP",
    "MUNICIPALITY_CODE",
    "ZIP_CODE",
    "TRADE_YEAR",
    "TRADE_MONTH",
    "TRADE_DAY",
    "NUMBER_ROOMS",
]
TARGET_COLUMN = "SQM_PRICE"

# -------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# -------------------------------------------------------------------
def transform_floor(x: str) -> int:
    """
    Convert floor strings to numeric.
    """
    try:
        return int(x)
    except (ValueError, TypeError):
        if isinstance(x, str):
            x_lower = x.lower()
            if x_lower in ["st"]:
                return 0
            elif x_lower in ["kl", "kld", "kælder"]:
                return -1
        return 0

def facility_clean(text: str) -> float:
    """
    Convert facility strings to 1 if not NaN, else 0.
    If you want more nuanced parsing (like "T:" -> toilet),
    you'd do that here. For now, just check notnull -> 1.
    """
    return float(pd.notna(text))

def groupby_mean_impute(df: pd.DataFrame, groupby_col: str, impute_col: str) -> pd.DataFrame:
    """
    For each group in `groupby_col`, fill the missing values in `impute_col` 
    with the group's mean. e.g. groupby ZIP_CODE => fill means per ZIP group.
    """
    df[impute_col] = df[impute_col].astype(float)
    df[impute_col] = df.groupby(groupby_col)[impute_col].transform(lambda x: x.fillna(x.mean()))
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mimic your reference approach:
      1) groupby_mean_impute for FLOOR, CONSTRUCTION_YEAR by ZIP_CODE
      2) if REBUILDING_YEAR is missing, set it to CONSTRUCTION_YEAR
      3) fill area columns with 0
      4) (NEW) fallback fill for any remaining NaNs
    """
    # 1) Group-based imputations
    if "ZIP_CODE" in df.columns:
        if "FLOOR" in df.columns:
            df = groupby_mean_impute(df, "ZIP_CODE", "FLOOR")
        if "CONSTRUCTION_YEAR" in df.columns:
            df = groupby_mean_impute(df, "ZIP_CODE", "CONSTRUCTION_YEAR")

    # 2) If REBUILDING_YEAR is missing, set to CONSTRUCTION_YEAR
    if "REBUILDING_YEAR" in df.columns and "CONSTRUCTION_YEAR" in df.columns:
        missing_rebuilding = df["REBUILDING_YEAR"].isna()
        df.loc[missing_rebuilding, "REBUILDING_YEAR"] = df.loc[missing_rebuilding, "CONSTRUCTION_YEAR"]

    # 3) Fill area columns with 0
    if set(AREA_COLUMNS).issubset(df.columns):
        df[AREA_COLUMNS] = df[AREA_COLUMNS].fillna(0)

    # 4) Fallback fill for any remaining NaNs
    #    If you still want a numeric ZIP_CODE, you could set them to -1:
    if "ZIP_CODE" in df.columns:
        df["ZIP_CODE"] = df["ZIP_CODE"].fillna(-1)

    #    Then fill numeric columns (e.g. FLOOR, CONSTRUCTION_YEAR, REBUILDING_YEAR) with their global means:
    if "FLOOR" in df.columns:
        df["FLOOR"].fillna(df["FLOOR"].mean(), inplace=True)

    if "CONSTRUCTION_YEAR" in df.columns:
        df["CONSTRUCTION_YEAR"].fillna(df["CONSTRUCTION_YEAR"].mean(), inplace=True)

    if "REBUILDING_YEAR" in df.columns:
        df["REBUILDING_YEAR"].fillna(df["REBUILDING_YEAR"].mean(), inplace=True)

    return df




# -------------------------------------------------------------------
# 2) FEATURE ENGINEERING
# -------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame, log_transform: bool = False) -> pd.DataFrame:
    """
    1) Parse TRADE_DATE into year/month/day
    2) Convert floor
    3) Convert facility columns to booleans
    4) Convert HAS_ELEVATOR to float
    5) Handle missing values via groupby mean, area fill
    6) (Optional) Log transform target
    """
    # --- 1) Parse date ---
    if "TRADE_DATE" in df.columns:
        df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"], errors="coerce")
        df["TRADE_YEAR"] = df["TRADE_DATE"].dt.year
        df["TRADE_MONTH"] = df["TRADE_DATE"].dt.month
        df["TRADE_DAY"] = df["TRADE_DATE"].dt.day
        df.drop("TRADE_DATE", axis=1, inplace=True)

    # --- 2) Floor parsing ---
    if "FLOOR" in df.columns:
        df["FLOOR"] = df["FLOOR"].apply(transform_floor).astype(float)
    else:
        df["FLOOR"] = 0.0

    # --- 3) Facility columns -> booleans ---
    # (or just notna() if your raw data means "string" => has facility)
    df["HAS_TOILET"] = df["FACILITIES_TOILET"].apply(facility_clean)
    df["HAS_SHOWER"] = df["FACILITIES_SHOWER"].apply(facility_clean)
    df["HAS_KITCHEN"] = df["FACILITIES_KITCHEN"].apply(facility_clean)

    # --- 4) HAS_ELEVATOR ---
    if "HAS_ELEVATOR" in df.columns:
        df["HAS_ELEVATOR"] = df["HAS_ELEVATOR"].map({"true": 1, "false": 0})
        df["HAS_ELEVATOR"] = df["HAS_ELEVATOR"].fillna(0).astype(float)
    else:
        df["HAS_ELEVATOR"] = 0.0

    # --- 5) Missing values handling (groupby mean, fill area, etc.) ---
    df = handle_missing_values(df)

    # --- 6) (Optional) Log transform the target (SQM_PRICE) ---
    if log_transform and TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = np.log1p(df[TARGET_COLUMN])  # log(1 + x)

    return df


# -------------------------------------------------------------------
# 3) PREPROCESS PIPELINE
# -------------------------------------------------------------------
def preprocess(
    raw_data_path: Path,
    output_folder: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    do_log_transform: bool = False
) -> None:
    """
    1) Load raw CSV
    2) Drop rows missing crucial 'PRICE' / 'SQM_PRICE'
    3) feature_engineering (floor, facility, date, handle missing)
    4) Drop columns in DROP_COLUMNS
    5) Split (train/val)
    6) Scale numeric features
    7) Save train_processed.csv, val_processed.csv, scaler.pkl
    """
    print("Starting data preprocessing...")
    df = pd.read_csv(raw_data_path)
    print(f"Initial shape: {df.shape}")

    # 1) Drop rows missing crucial columns
    df = df.dropna(subset=["PRICE", "SQM_PRICE"])

    # 2) Feature engineering
    df = feature_engineering(df, log_transform=do_log_transform)

    # 3) Drop columns we don’t need
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True, errors="ignore")

    # 4) Split into train / val
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    # 5) Scale numeric columns
    numeric_cols_to_scale = []
    for col in FEATURE_COLUMNS:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols_to_scale.append(col)

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols_to_scale])  # fit on train only

    train_df[numeric_cols_to_scale] = scaler.transform(train_df[numeric_cols_to_scale])
    val_df[numeric_cols_to_scale]   = scaler.transform(val_df[numeric_cols_to_scale])

    # 6) Save outputs
    output_folder.mkdir(parents=True, exist_ok=True)
    train_out = output_folder / "train_processed.csv"
    val_out   = output_folder / "val_processed.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"Saved train to: {train_out}")
    print(f"Saved val to:   {val_out}")

    scaler_path = output_folder / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    print("Preprocessing complete.")


# -------------------------------------------------------------------
# 4) PRICE DATASET
# -------------------------------------------------------------------
class PriceDataset(Dataset):
    """
    A custom PyTorch Dataset that loads a *processed* CSV
    (already feature-engineered & scaled)
    and returns (features, target) as torch.Tensors.
    """

    def __init__(self, csv_path: Path, train: bool = True) -> None:
        self.csv_path = csv_path
        print(f"Loading data from: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)

        # If 'train=True' but we can't find the target column => error
        if train and (TARGET_COLUMN not in self.data.columns):
            raise ValueError(f"Target column {TARGET_COLUMN} not found in {csv_path}")

        if train:
            self.targets = self.data[TARGET_COLUMN].values.astype("float32")
        else:
            self.targets = None

        for col in FEATURE_COLUMNS:
            if col not in self.data.columns:
                print(f"Warning: Feature '{col}' not found, filling with 0.")
                self.data[col] = 0.0

        self.features = self.data[FEATURE_COLUMNS].values.astype("float32")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            return x
