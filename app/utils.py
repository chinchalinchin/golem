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

logger = logging.getLogger(__name__)

COMMAND_REGISTRY = {}

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

def get_vizdoom_game(pth: str, scenario: str, sensors=None, mode=vizdoom.Mode.PLAYER) -> vizdoom.DoomGame:
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
    game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    
    # Dynamically set the mode (Defaults to PLAYER for inference, SPECTATOR for recording)
    game.set_mode(mode) 
    
    # Enforce HUD rendering to satisfy the 'Doomguy' heuristic requirement
    game.set_render_hud(True)
    
    if sensors:
        logger.info(f"Configuring Sensors - Depth: {getattr(sensors, 'depth', False)}, Audio: {getattr(sensors, 'audio', False)}")
        if getattr(sensors, 'depth', False):
            game.set_depth_buffer_enabled(True)
        if getattr(sensors, 'audio', False):
            game.set_audio_buffer_enabled(True)
            
    return game

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

def get_latest_parameters(archives: List[str]) -> Tuple[int, int]:
    """
    Find the latest parameters of the trained model.
    """
    cortical_depth, working_memory = None, None
    if archives:
        latest_archive = sorted(archives, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        try:
            parts = latest_archive.name.split('.')
            for part in parts:
                if part.startswith('c-'):
                    cortical_depth = int(part[2:])
                elif part.startswith('w-'):
                    working_memory = int(part[2:])
            logger.info(f"Discovered brain architecture from {latest_archive.name}: depth={cortical_depth}, memory={working_memory}")
        except Exception as e:
            logger.warning(f"Failed to parse architecture from {latest_archive.name}: {e}")
    return cortical_depth, working_memory