# Standard Libraries
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from enum import Enum

# External Libraries
import yaml
from pydantic import BaseModel

# Application Libraries
from app.utils.conf import resolve_path

logger = logging.getLogger(__name__)

class LossType(str, Enum):
    FOCAL = "focal"
    BCE = "bce"
    SMOOTH = "smooth"
    ASL = "asl"

class ObligeConfig(BaseModel):
    # Allow properties to be defined as static strings or lists for randomized ranges
    game: Union[str, List[str]] = "doom2"
    engine: Union[str, List[str]] = "zdoom"
    length: Union[str, List[str]] = "single"
    theme: Union[str, List[str]] = "original"
    size: Union[str, List[str]] = "small"
    outdoors: Union[str, List[str]] = "mixed"
    caves: Union[str, List[str]] = "mixed"
    liquids: Union[str, List[str]] = "none"
    hallways: Union[str, List[str]] = "mixed"
    teleporters: Union[str, List[str]] = "none"
    steepness: Union[str, List[str]] = "mixed"
    mons: Union[str, List[str]] = "normal"
    strength: Union[str, List[str]] = "medium"
    health: Union[str, List[str]] = "normal"
    ammo: Union[str, List[str]] = "normal"
    weapons: Union[str, List[str]] = "normal"

class RandomizerConfig(BaseModel):
    executable: str
    output: str
    iterations: int = 5
    duration: int = 60
    oblige: ObligeConfig

class ModuleConfig(BaseModel):
    scenario: str
    episodes: int = 5
    map: Optional[str] = "map01" 

class DataConfig(BaseModel):
    prefix: str
    dirs: Dict[str, str]

class AppConfig(BaseModel):
    name: str
    version: str
    log_level: str

class AugmentationConfig(BaseModel):
    mirror: bool = False

class SensorsConfig(BaseModel):
    visual: bool = True
    depth: bool = False
    audio: bool = False
    thermal: bool = False

class DSPConfig(BaseModel):
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64

class BrainConfig(BaseModel):
    mode: str
    cortical_depth: int = 2
    working_memory: int = 64
    activation: float = 0.5
    sensors: SensorsConfig = SensorsConfig()
    dsp: DSPConfig = DSPConfig()

class FocalConfig(BaseModel):
    alpha: float = 0.25
    gamma: float = 2.0

class AsymmetricConfig(BaseModel):
    gamma_pos: float = 1.0
    gamma_neg: float = 4.0
    clip: float = 0.05

class LabelSmoothingConfig(BaseModel):
    epsilon: float = 0.1

class LossConfig(BaseModel): 
    focal: FocalConfig
    asymmetric: AsymmetricConfig
    smooth: LabelSmoothingConfig

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    sequence_length: int = 32
    action_space_size: int = 8 
    action_names: List[str] = [] 
    loss: LossType = LossType.FOCAL    
    augmentation: AugmentationConfig = AugmentationConfig()

class GolemConfig(BaseModel):
    app: AppConfig
    config: Dict[str, str]
    keybindings: Dict[str, Dict[str, str]]
    data: DataConfig
    training: TrainingConfig
    brain: BrainConfig
    loss: LossConfig
    randomizer: RandomizerConfig
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
        
        active_profile = cfg.brain.mode
        if active_profile not in cfg.config:
            raise ValueError(f"Active profile '{active_profile}' not found in the 'config' block.")
            
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