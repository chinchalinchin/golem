"""
Utility functions for path resolution, logging setup, and resource location.
"""
# Standard Libraries
import os
import logging
from pathlib import Path
from typing import Callable, Union, List, Tuple

# External Libraries
import vizdoom
import cv2
import torch
import numpy as np

logger = logging.getLogger(__name__)

COMMAND_REGISTRY = {}

MODEL_ARCHIVE_TEMPLATE = "{date}.c-{c}.w-{w}.v-{v}.d-{d}.a-{a}.t-{t}.sr-{sr}.nf-{nf}.hl-{hl}.nm-{nm}"

# ----------------------------------------------------------------------------
# --------------------------------------- APPLICATION UTILS
# ----------------------------------------------------------------------------

def setup_logging(level_str: str = "INFO"):
    """Configures the root logger with a standard format."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def register_command(name: str = None) -> Callable:
    """
    Decorator to register a CLI command into the global registry.
    """
    def decorator(func: Callable) -> Callable:
        cmd_name = name if name else func.__name__
        COMMAND_REGISTRY[cmd_name] = func
        return func
    return decorator

# ----------------------------------------------------------------------------
# --------------------------------------- FILE UTILS
# ----------------------------------------------------------------------------

def get_project_root() -> Path:
    """Returns the absolute path to the project root (one level up from app)."""
    return Path(__file__).parent.parent.resolve()

def resolve_path(path_str: str) -> str:
    """
    Resolves a path relative to the project root.
    If the path is already absolute, it is returned as-is.
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
        
    root = get_project_root()
    return str(root / path)

def get_unique_filename(directory: Union[str, Path], prefix: str, extension: str = "npz") -> str:
    """Generates a unique filename (e.g., data_1.npz) to prevent overwrites."""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = Path(resolve_path(str(directory)))
        
    directory.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        filename = f"{prefix}.{counter}.{extension}"
        full_path = directory / filename
        if not full_path.exists():
            return str(full_path)
        counter += 1

# ----------------------------------------------------------------------------
# --------------------------------------- VIZDOOM UTILS
# ----------------------------------------------------------------------------

def get_vizdoom_scenario(scenario_name: str) -> str:
    """
    Locates a ViZDoom scenario WAD file.
    """
    if os.path.basename(scenario_name) == scenario_name:
        package_path = os.path.dirname(vizdoom.__file__)
        scenario_path = os.path.join(package_path, "scenarios", scenario_name)
        if os.path.exists(scenario_path):
            return scenario_path

    local_path = resolve_path(scenario_name)
    if os.path.exists(local_path):
        return local_path
        
    package_path = os.path.dirname(vizdoom.__file__)
    scenario_path = os.path.join(package_path, "scenarios", scenario_name)
    logger.warning(f"Could not find scenario. Checked:\n  {scenario_path}\n  {local_path}")
        
    return scenario_path


def get_vizdoom_game(pth: str, scenario: str, sensors=None, mode=vizdoom.Mode.PLAYER, map_name=None) -> vizdoom.DoomGame:
    """
    Retrieves a ViZDoom DoomGame instances configured for Golem training.
    """
    cfg_path = resolve_path(pth)
    scenario_path = get_vizdoom_scenario(scenario)
    
    logger.info(f"Loading ViZDoom Config: {cfg_path}")
    logger.info(f"Loading ViZDoom Scenario: {scenario_path}")
    logger.info(f"Initializing Engine in Mode: {mode.name}")
    
    game = vizdoom.DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    
    # Inject the runtime map override if provided
    if map_name:
        game.set_doom_map(map_name)
        
    game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    
    # Dynamically set the mode (Defaults to PLAYER for inference, SPECTATOR for recording)
    game.set_mode(mode) 
    
    # Enforce HUD rendering to satisfy the 'Doomguy' heuristic requirement
    game.set_render_hud(True)
    
    if sensors:
        logger.info(f"Configuring Sensors - Depth: {getattr(sensors, 'depth', False)}, Audio: {getattr(sensors, 'audio', False)}, Thermal: {getattr(sensors, 'thermal', False)}")
        
        if getattr(sensors, 'depth', False):
            game.set_depth_buffer_enabled(True)
        if getattr(sensors, 'audio', False):
            game.set_audio_buffer_enabled(True)
        if getattr(sensors, 'thermal', False):
            game.set_labels_buffer_enabled(True)  
    return game

# ----------------------------------------------------------------------------
# --------------------------------------- MODEL UTILS
# ----------------------------------------------------------------------------

def get_latest_parameters(archives: List[Path]) -> dict:
    """
    Find the latest parameters of the trained model from the filename.
    """
    params = {}
    if archives:
        latest_archive = sorted(archives, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        try:
            parts = latest_archive.name.split('.')
            for part in parts:
                if part.startswith('c-'): params['c'] = int(part[2:])
                elif part.startswith('w-'): params['w'] = int(part[2:])
                elif part.startswith('v-'): params['v'] = bool(int(part[2:]))
                elif part.startswith('d-'): params['d'] = bool(int(part[2:]))
                elif part.startswith('a-'): params['a'] = bool(int(part[2:]))
                elif part.startswith('t-'): params['t'] = bool(int(part[2:]))
                elif part.startswith('sr-'): params['sr'] = int(part[3:])
                elif part.startswith('nf-'): params['nf'] = int(part[3:])
                elif part.startswith('hl-'): params['hl'] = int(part[3:])
                elif part.startswith('nm-'): params['nm'] = int(part[3:])
            logger.info(f"Discovered brain architecture from {latest_archive.name}: {params}")
        except Exception as e:
            logger.warning(f"Failed to parse architecture from {latest_archive.name}: {e}")
    return params

def apply_latest_parameters(cfg, archives: List[Path]) -> Tuple[int, int]:
    """Helper to parse and apply filename parameters to the active config."""
    c, w = cfg.brain.cortical_depth, cfg.brain.working_memory
    params = get_latest_parameters(archives)
    if params:
        c = params.get('c', c)
        w = params.get('w', w)
        cfg.brain.sensors.visual = params.get('v', cfg.brain.sensors.visual)
        cfg.brain.sensors.depth = params.get('d', cfg.brain.sensors.depth)
        cfg.brain.sensors.audio = params.get('a', cfg.brain.sensors.audio)
        cfg.brain.sensors.thermal = params.get('t', cfg.brain.sensors.thermal)
        cfg.brain.dsp.sample_rate = params.get('sr', cfg.brain.dsp.sample_rate)
        cfg.brain.dsp.n_fft = params.get('nf', cfg.brain.dsp.n_fft)
        cfg.brain.dsp.hop_length = params.get('hl', cfg.brain.dsp.hop_length)
        cfg.brain.dsp.n_mels = params.get('nm', cfg.brain.dsp.n_mels)
    return c, w

def normalize_audio_buffer(raw_audio: np.ndarray) -> np.ndarray:
    """Applies zero-mean, unit-variance normalization to a raw engine audio buffer."""
    mean = np.mean(raw_audio, axis=-1, keepdims=True)
    std = np.std(raw_audio, axis=-1, keepdims=True) + 1e-8
    return (raw_audio - mean) / std

# ----------------------------------------------------------------------------
# --------------------------------------- CLASSES
# ----------------------------------------------------------------------------

class SensoryExtractor:
    """
    Centralized utility class for extracting, normalizing, and formatting 
    ViZDoom phenomenological buffers into numpy arrays and PyTorch tensors.
    """
    
    @staticmethod
    def get_numpy_state(state, sensors_cfg) -> dict:
        """Extracts and normalizes raw game state buffers into a dictionary."""
        data = {}
        
        # 1. Visual (RGB)
        if state.screen_buffer is not None:
            data['visual'] = cv2.resize(state.screen_buffer.transpose(1, 2, 0), (64, 64)) / 255.0
            
        # 2. Depth
        if getattr(sensors_cfg, 'depth', False) and state.depth_buffer is not None:
            data['depth'] = cv2.resize(state.depth_buffer, (64, 64)) / 255.0
            
        # 3. Audio (Normalized Raw Waveform)
        if getattr(sensors_cfg, 'audio', False) and state.audio_buffer is not None:
            raw_audio = state.audio_buffer
            mean = np.mean(raw_audio, axis=-1, keepdims=True)
            std = np.std(raw_audio, axis=-1, keepdims=True) + 1e-8
            data['audio'] = (raw_audio - mean) / std
            
        # 4. Thermal
        if getattr(sensors_cfg, 'thermal', False) and state.labels_buffer is not None:
            binary_mask = (state.labels_buffer > 0).astype(np.float32)
            data['thermal'] = cv2.resize(binary_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
            
        return data

    @staticmethod
    def to_tensors(numpy_state: dict, device: torch.device) -> dict:
        """Converts normalized numpy dictionary into PyTorch tensors for model inference."""
        tensors = {}
        
        # Visual & Depth Concatenation
        if 'visual' in numpy_state:
            x_vis_np = numpy_state['visual']
            if 'depth' in numpy_state:
                depth_expanded = np.expand_dims(numpy_state['depth'], axis=2)
                x_vis_np = np.concatenate((x_vis_np, depth_expanded), axis=2)
            tensors['visual'] = torch.from_numpy(np.transpose(x_vis_np, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            
        # Audio (Now passes the raw 1D waveform directly to the model)
        if 'audio' in numpy_state:
            tensors['audio'] = torch.from_numpy(numpy_state['audio']).float().unsqueeze(0).unsqueeze(0).to(device)
            
        # Thermal Mask
        if 'thermal' in numpy_state:
            tensors['thermal'] = torch.from_numpy(numpy_state['thermal']).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        
        return tensors