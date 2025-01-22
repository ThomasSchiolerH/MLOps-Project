# Use an official PyTorch image as base (CPU version for now)
#FROM pytorch/pytorch:latest
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy data
#COPY data /app/data

COPY data.dvc /app/data.dvc

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true

# Copy SRC
COPY src /app/src

# Default command to run the training script
CMD ["sh", "-c", "dvc pull && python src/AVM/main.py train"]

