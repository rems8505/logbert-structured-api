from pydantic_settings import BaseSettings
from pathlib import Path

class BertSettings(BaseSettings):
    # Model paths
    tokenizer_path: str = "."
    tokenizer_file: str = "tokenizer.json"
    model_config_file: str = "config.json"
    model_weights_file: str = "vhm_center.pt"
    
    # Model parameters
    max_length: int = 512
    
    class Config:
        case_sensitive = True

bert_settings = BertSettings()