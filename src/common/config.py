import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DataCollection:
    PDF_FOLDER: str = os.getenv("PDF_FOLDER", str(Path("data/pdfs")))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(Path("output_data")))
    WEB_SCRAPING_URLS: List[str] = os.getenv("WEB_SCRAPING_URLS", "https://afdc.energy.gov/fuels/electricity_charging_home.html,https://www.energy.gov/eere/electricvehicles/electric-vehicles-charging,https://www.chargepoint.com/drivers/charging-basics").split(",")
    EV_KEYWORDS: List[str] = os.getenv("EV_KEYWORDS", "electric vehicle,charging,charger,EV,battery,plug-in,station").split(",")
    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "50"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "0.3"))
    DELAY_BETWEEN_REQUESTS: float = float(os.getenv("DELAY_BETWEEN_REQUESTS", "2.0"))

@dataclass
class Model:
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "facebook/opt-1.3b")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(Path("output_data")))
    MODEL_DIR: str = os.getenv("MODEL_DIR", str(Path("models/fine_tuned_model")))
    LORA_R: int = int(os.getenv("LORA_R", "4"))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "8"))
    LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", "0.1"))
    TRAIN_BATCH_SIZE: int = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
    EVAL_BATCH_SIZE: int = int(os.getenv("EVAL_BATCH_SIZE", "4"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    QUANTIZATION_ENABLED: bool = os.getenv("QUANTIZATION_ENABLED", "False").lower() == "true"
    QUANTIZATION_TYPE: str = os.getenv("QUANTIZATION_TYPE", "nf4")

@dataclass
class Security:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")

@dataclass
class API:
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))

@dataclass
class MLflow:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ev-charging-llm")

@dataclass
class Config:
    DataCollection: DataCollection = DataCollection()
    Model: Model = Model()
    Security: Security = Security()
    API: API = API()
    MLflow: MLflow = MLflow()

config = Config()