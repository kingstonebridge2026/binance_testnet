FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only Torch first (This is the big space saver)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the code
COPY . .

CMD ["python", "Sniper.py"]
