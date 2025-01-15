import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Path to the processed dataset
PROCESSED_FILE = "../../data/processed/HACKATHON.AVM_EJERLEJLIGHEDER_TRAIN.csv"

# Load the processed data
data = pd.read_csv(PROCESSED_FILE)

# Print the first few rows for debugging
print("First 5 rows of the dataset:")
print(data.head())

# Drop rows with missing target values
data = data.dropna(subset=["SQM_PRICE"])

# Fill or drop other missing values
data = data.fillna(0)

# Encode categorical variables (if any)
# data["HAS_ELEVATOR"] = data["HAS_ELEVATOR"].astype(int)

# Select features and target
features = [
    "AREA_TINGLYST", "AREA_RESIDENTIAL", "NUMBER_ROOMS", "DISTANCE_LAKE", "DISTANCE_HARBOUR", "DISTANCE_COAST","CONSTRUCTION_YEAR"
]
X = data[features]
y = data["SQM_PRICE"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")