FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory to `src` for command execution
WORKDIR /app/src

# Default command (can be overridden when running the container)
CMD ["python", "-m", "AVM.main"]
