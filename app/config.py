import yaml
import logging
import re
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional
from app.utils import resolve_path

logger = logging.getLogger(__name__)

class ModuleConfig(BaseModel):
    scenario: str
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

class BrainConfig(BaseModel):
    cortical_depth: int = 2
    working_memory: int = 64

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    model_save_path: str
    sequence_length: int = 32
    config: str  # e.g., "fluid"
    action_space_size: int = 8 
    action_names: List[str] = [] 
    augmentation: AugmentationConfig = AugmentationConfig()

class GolemConfig(BaseModel):
    app: AppConfig
    config: Dict[str, str] # NEW: Maps profile name to .cfg path
    keybindings: Dict[str, Dict[str, str]] # NEW: Nested profile mappings
    data: DataConfig
    training: TrainingConfig
    brain: BrainConfig
    modules: Dict[str, ModuleConfig]

    @classmethod
    def load(cls, config_path: str = "conf/app.yaml") -> "GolemConfig":
        full_path = resolve_path(config_path)
        path = Path(full_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")
        
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        cfg = cls(**raw_config)
        
        # Determine the active profile
        active_profile = cfg.training.config
        if active_profile not in cfg.config:
            raise ValueError(f"Active profile '{active_profile}' not found in the 'config' block.")
            
        # Parse the corresponding ViZDoom .cfg 
        vizdoom_cfg_path = resolve_path(cfg.config[active_profile])
        try:
            with open(vizdoom_cfg_path, "r") as f:
                content = f.read()
                
            match = re.search(r'available_buttons\s*=\s*\{([^}]+)\}', content)
            if match:
                buttons = match.group(1).split()
                cfg.training.action_names = [b.strip() for b in buttons if b.strip()]
                cfg.training.action_space_size = len(cfg.training.action_names)
            else:
                logger.warning("Could not parse available_buttons from config.")
        except Exception as e:
            logger.error(f"Failed to parse ViZDoom config: {e}")
            
        return cfg