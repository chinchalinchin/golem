# Standard Libraries
import os
import logging

# External Libraries
import vizdoom

# Application Libraries
from app.utils.conf import resolve_path


logger = logging.getLogger(__name__)


def get_scenario(scenario_name: str) -> str:
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


def get_game(pth: str, scenario: str, sensors=None, mode=vizdoom.Mode.PLAYER, map_name=None) -> vizdoom.DoomGame:
    """
    Retrieves a ViZDoom DoomGame instances configured for Golem training.
    """
    cfg_path = resolve_path(pth)
    scenario_path = get_scenario(scenario)
    
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

