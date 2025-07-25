FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=1000

COPY src/ .

CMD ["python", "src/main.py"]