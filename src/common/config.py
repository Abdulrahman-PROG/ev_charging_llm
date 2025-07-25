import os
from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class DataCollection:
    PDF_FOLDER: str = str(Path("data/pdfs"))
    OUTPUT_DIR: str = str(Path("output_data"))
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", 100))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", 10000))
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", 0.3))
    EV_KEYWORDS: List[str] = ["electric vehicle", "EV charging", "charging station", 
                             "battery", "plug-in", "charger", "kilowatt", "range"]
    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; EVChargingBot/1.0)")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 10))
    DELAY_BETWEEN_REQUESTS: float = float(os.getenv("DELAY_BETWEEN_REQUESTS", 2.0))
    WEB_SCRAPING_URLS: List[str] = os.getenv(
        "WEB_SCRAPING_URLS", 
        "https://afdc.energy.gov/fuels/electricity_charging_home.html,"
        "https://www.energy.gov/eere/electricvehicles/electric-vehicles-charging,"
        "https://www.chargepoint.com/drivers/charging-basics"
    ).split(",")

@dataclass
class FineTuning:
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "facebook/opt-1.3b")
    OUTPUT_DIR: str = str(Path("models/fine_tuned_model"))
    LORA_R: int = int(os.getenv("LORA_R", 16))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", 32))
    LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", 0.05))
    TRAIN_BATCH_SIZE: int = int(os.getenv("TRAIN_BATCH_SIZE", 4))
    EVAL_BATCH_SIZE: int = int(os.getenv("EVAL_BATCH_SIZE", 4))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 2e-4))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", 3))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))
    QUANTIZATION_ENABLED: bool = os.getenv("QUANTIZATION_ENABLED", "False").lower() == "true"
    QUANTIZATION_TYPE: str = os.getenv("QUANTIZATION_TYPE", "nf4")

@dataclass
class Evaluation:
    OUTPUT_DIR: str = str(Path("evaluation_results"))
    EVAL_BATCH_SIZE: int = int(os.getenv("EVAL_BATCH_SIZE", 4))
    MAX_SAMPLES: int = int(os.getenv("MAX_SAMPLES", 100))
    METRICS: List[str] = ["ROUGE", "BLEU", "inference_time", "perplexity"]

@dataclass
class Serving:
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_WORKERS: int = int(os.getenv("API_WORKERS", 4))
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", 9090))
    MAX_REQUESTS: int = int(os.getenv("MAX_REQUESTS", 100))
    TIMEOUT: int = int(os.getenv("TIMEOUT", 60))

@dataclass
class Security:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")

@dataclass
class MLflow:
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ev-charging-llm")

@dataclass
class Config:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DATA_COLLECTION: DataCollection = DataCollection()
    FINE_TUNING: FineTuning = FineTuning()
    EVALUATION: Evaluation = Evaluation()
    SERVING: Serving = Serving()
    SECURITY: Security = Security()
    MLFLOW: MLflow = MLflow()

config = Config()