FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for torch and other libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=2000

COPY src/ .

CMD ["python", "serving/serve.py"]