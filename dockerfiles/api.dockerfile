# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

EXPOSE $PORT

# Create a working directory
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*

COPY src /app/src

COPY data/processed/scaler.pkl /app/data/processed/scaler.pkl

EXPOSE 8080

ENTRYPOINT ["uvicorn", "src.AVM.api:app", "--host", "0.0.0.0", "--port", "8080"]

#CMD exec uvicorn src.AVM.api:app --port $PORT --host 0.0.0.0 --workers 1

