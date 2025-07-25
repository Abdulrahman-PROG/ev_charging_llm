import os
import time
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORS
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prometheus_client import Counter, Histogram, start_http_server
import psutil
from common.logging import setup_logging
from common.utils import format_prompt, generate_response
from config import config

# Setup logging
logger = setup_logging(__name__)

# Setup Prometheus
if config.Monitoring.PROMETHEUS_ENABLED:
    start_http_server(config.Monitoring.METRICS_PORT)
request_counter = Counter("api_requests_total", "Total API requests")
request_latency = Histogram("api_request_latency_seconds", "Request latency")

# Setup FastAPI
app = FastAPI()
CORS(app, origins=config.Deployment.CORS_ORIGINS, allow_methods=config.Deployment.CORS_METHODS, allow_headers=config.Deployment.CORS_HEADERS)

# Authentication
api_key_header = APIKeyHeader(name="X-API-Key")
def verify_api_key(api_key: str = Depends(api_key_header)):
    if config.Security.API_KEY_REQUIRED and api_key != config.Security.SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Model Server
class ModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
    
    def load_model(self):
        start_time = time.time()
        try:
            model_path = config.Model.FINAL_MODEL_PATH
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.Model.QUANTIZATION_TYPE,
                bnb_4bit_compute_dtype=torch.float16
            )
            if os.path.exists(model_path):
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
            else:
                logger.warning(f"Fine-tuned model not found, using base model")
                self.tokenizer = AutoTokenizer.from_pretrained(config.Model.BASE_MODEL_NAME, token=config.Model.HF_TOKEN)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.Model.BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=config.Model.HF_TOKEN
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def get_system_metrics(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }

model_server = ModelServer()

class GenerateRequest(BaseModel):
    instruction: str
    input: str = ""
    max_length: int = config.Model.GENERATION_MAX_LENGTH
    temperature: float = config.Model.GENERATION_TEMPERATURE

@app.on_event("startup")
async def startup_event():
    model_server.load_model()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_server.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": config.PROJECT_VERSION
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "system_metrics": model_server.get_system_metrics(),
        "config": config.get_config_summary()
    }

@app.post("/generate")
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    with request_latency.time():
        request_counter.inc()
        try:
            if not 0.1 <= request.temperature <= 1.0:
                raise HTTPException(status_code=400, detail="Temperature must be between 0.1 and 1.0")
            if not 50 <= request.max_length <= 500:
                raise HTTPException(status_code=400, detail="Max length must be between 50 and 500")
            prompt = format_prompt(request.instruction, request.input)
            response = generate_response(model_server.model, model_server.tokenizer, prompt, request.max_length, request.temperature)
            return {"response": response, "instruction": request.instruction, "input": request.input}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.Deployment.API_HOST,
        port=config.Deployment.API_PORT,
        workers=config.Deployment.API_WORKERS
    )