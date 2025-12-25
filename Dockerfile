
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install only essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-optimized AI libraries to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code
COPY . .

CMD ["python", "Sniper.py"]








