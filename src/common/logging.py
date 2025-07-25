import logging
import os
from common.config import config
def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging configuration for a module"""
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, config.Monitoring.LOG_LEVEL))
    
    # Create handlers
    file_handler = logging.FileHandler(config.Monitoring.LOG_FILE)
    stream_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(config.Monitoring.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger