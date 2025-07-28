import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Server configuration
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 8000))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model configuration
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv('HUGGINGFACE_TOKEN')
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'your-username/your-model-name')
    
    # Database configuration
    CSV_PATH: str = os.getenv('CSV_PATH', 'data/df_obat.csv')
    
    # API configuration
    MAX_TOP_K: int = int(os.getenv('MAX_TOP_K', 20))
    DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', 5))
    
    # Initialization configuration
    AUTO_INITIALIZE: bool = os.getenv('AUTO_INITIALIZE', 'True').lower() == 'true'
    STARTUP_TIMEOUT: int = int(os.getenv('STARTUP_TIMEOUT', 600))  # 10 minutes
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')