FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "serving.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]