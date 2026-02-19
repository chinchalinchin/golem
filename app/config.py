import yaml
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Optional
from app.utils import resolve_path

logger = logging.getLogger(__name__)

class ModuleConfig(BaseModel):
    scenario: str
    config: str
    episodes: int = 5

class DataConfig(BaseModel):
    output_dir: str
    filename_prefix: str

class AppConfig(BaseModel):
    name: str
    version: str
    log_level: str

class AugmentationConfig(BaseModel):
    mirror: bool = False

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    model_save_path: str
    sequence_length: int = 32
    action_space_size: int = 8
    augmentation: AugmentationConfig = AugmentationConfig()

class GolemConfig(BaseModel):
    app: AppConfig
    keybindings: Dict[str, str]
    data: DataConfig
    training: TrainingConfig
    modules: Dict[str, ModuleConfig]

    @classmethod
    def load(cls, config_path: str = "conf/app.yaml") -> "GolemConfig":
        full_path = resolve_path(config_path)
        path = Path(full_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")
        
        logger.debug(f"Loading configuration from {path}")
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        return cls(**raw_config)