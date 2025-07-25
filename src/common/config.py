import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

@dataclass
class DataCollection:
    WEB_SCRAPING_URLS: list = [url.strip() for url in os.getenv('WEB_SCRAPING_URLS', '').split(',') if url.strip()]
    PDF_DIR: str = os.path.join(DATA_DIR, 'pdfs')
    OUTPUT_DIR: str = OUTPUT_DIR
    MIN_TEXT_LENGTH: int = 50
    EV_KEYWORDS: list = ['electric vehicle', 'EV', 'charging', 'charger', 'battery', 'station']

@dataclass
class Model:
    BASE_MODEL_NAME: str = os.getenv('BASE_MODEL_NAME', 'facebook/opt-1.3b')
    HF_TOKEN: str = os.getenv('HF_TOKEN', '')
    FINAL_MODEL_PATH: str = os.path.join(MODEL_DIR, 'fine_tuned_model', 'final_model')
    QUANTIZATION_ENABLED: bool = os.getenv('QUANTIZATION_ENABLED', 'False').lower() == 'true'
    QUANTIZATION_TYPE: str = os.getenv('QUANTIZATION_TYPE', 'nf4')
    TRAIN_BATCH_SIZE: int = 4
    EVAL_BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    WARMUP_STEPS: int = 500
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.1
    GENERATION_MAX_LENGTH: int = 200
    GENERATION_TEMPERATURE: float = 0.7

@dataclass
class Logging:
    LOG_DIR: str = LOG_DIR

@dataclass
class Monitoring:
    PROMETHEUS_ENABLED: bool = os.getenv('PROMETHEUS_ENABLED', 'False').lower() == 'true'
    METRICS_PORT: int = int(os.getenv('METRICS_PORT', 9090))

@dataclass
class Security:
    API_KEY_REQUIRED: bool = os.getenv('API_KEY_REQUIRED', 'True').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', '')
    JWT_SECRET: str = os.getenv('JWT_SECRET', '')
    JWT_ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

@dataclass
class Deployment:
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', 8000))
    API_WORKERS: int = int(os.getenv('API_WORKERS', 4))
    CORS_ORIGINS: list = ['*']
    CORS_METHODS: list = ['*']
    CORS_HEADERS: list = ['*']

@dataclass
class Config:
    DataCollection: DataCollection = DataCollection()
    Model: Model = Model()
    Logging: Logging = Logging()
    Monitoring: Monitoring = Monitoring()
    Security: Security = Security()
    Deployment: Deployment = Deployment()
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    MLFLOW_TRACKING_URI: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_EXPERIMENT_NAME: str = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ev-charging-llm')
    PROJECT_VERSION: str = '1.0.0'

    def get_config_summary(self):
        return {
            "environment": self.ENVIRONMENT,
            "model": {
                "base_model": self.Model.BASE_MODEL_NAME,
                "quantization_enabled": self.Model.QUANTIZATION_ENABLED
            },
            "deployment": {
                "api_host": self.Deployment.API_HOST,
                "api_port": self.Deployment.API_PORT
            }
        }

config = Config()