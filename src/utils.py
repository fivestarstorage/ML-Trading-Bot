import logging
import yaml
import os
import pandas as pd

def load_config(config_path="config.yml"):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        # Fallback to looking in root if path is relative
        root_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        if os.path.exists(root_config):
            config_path = root_config
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir="logs", log_name="trading_bot"):
    """Setup logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(log_name)

def get_logger(name="trading_bot"):
    return logging.getLogger(name)


