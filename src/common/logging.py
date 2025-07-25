import logging
import os
from datetime import datetime
from common.config import config

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Use INFO as default log level
    
    log_dir = config.Logging.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger