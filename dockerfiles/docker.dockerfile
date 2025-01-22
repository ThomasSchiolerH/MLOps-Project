FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Copy the data folder into the container
COPY data /app/data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can be overridden when running the container)
CMD ["python", "-m", "src.AVM.main", "evaluate", "--model-checkpoint", "models/price_model.pth", "--test-file", "data/processed/test_processed.csv"]
