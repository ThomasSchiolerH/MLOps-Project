# Use an official PyTorch image as base (CPU version for now)
#FROM pytorch/pytorch:latest
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY src /app/src


# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Default command to run the training script
CMD ["python", "src/AVM/main.py", "train"]
