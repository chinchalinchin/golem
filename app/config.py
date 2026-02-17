import yaml
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

class VizDoomConfig(BaseModel):
    config_path: str
    scenario_name: str
    resolution: str
    episodes: int = 5

class DataConfig(BaseModel):
    output_dir: str
    filename_prefix: str

class AppConfig(BaseModel):
    name: str
    version: str
    log_level: str

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    model_save_path: str
    sequence_length: int = 32

class GolemConfig(BaseModel):
    app: AppConfig
    vizdoom: VizDoomConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def load(cls, config_path: str = "conf/app.yaml") -> "GolemConfig":
        """Loads the YAML configuration into a Pydantic model."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")
        
        logger.debug(f"Loading configuration from {path}")
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        return cls(**raw_config)