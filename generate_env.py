import os
import secrets

def generate_env_file():
    """Generate .env file with default values"""
    env_content = f"""ENVIRONMENT=development
BASE_MODEL_NAME=facebook/opt-1.3b
HF_TOKEN=
SECRET_KEY={secrets.token_hex(32)}
JWT_SECRET={secrets.token_hex(32)}
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=ev-charging-llm
WEB_SCRAPING_URLS=https://afdc.energy.gov/fuels/electricity_charging_home.html,https://www.energy.gov/eere/electricvehicles/electric-vehicles-charging,https://www.chargepoint.com/drivers/charging-basics
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
METRICS_PORT=9090
QUANTIZATION_ENABLED=False
QUANTIZATION_TYPE=nf4
"""
    with open(".env", "w") as f:
        f.write(env_content)
    print("Generated .env file successfully")

if __name__ == "__main__":
    if not os.path.exists(".env"):
        generate_env_file()
    else:
        print(".env file already exists")