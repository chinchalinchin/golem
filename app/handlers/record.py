import logging
import cv2
import numpy as np
from pathlib import Path
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from app.models.config import GolemConfig
from app.utils import resolve_path, get_unique_filename, get_vizdoom_scenario

logger = logging.getLogger(__name__)

def record_data(cfg: GolemConfig, module_name: str = "basic"):
    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found. Available: {list(cfg.modules.keys())}")
        return

    module = cfg.modules[module_name]
    active_profile = cfg.brain.mode
    
    if active_profile not in cfg.config:
        logger.error(f"Profile '{active_profile}' not found in config mapping.")
        return
        
    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    prefix_clean = cfg.data.prefix.rstrip('_')
    file_prefix = f"{prefix_clean}_{module_name}"
    output_path = get_unique_filename(output_dir, file_prefix, "npz")
    
    cfg_path = resolve_path(cfg.config[active_profile])
    scenario_path = get_vizdoom_scenario(module.scenario)

    logger.info(f"--- Recording Module: {module_name} ---")
    
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_doom_scenario_path(scenario_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    game.init()
    
    active_bindings = cfg.keybindings.get(active_profile, {})
    for key, command in active_bindings.items():
        game.send_game_command(f"bind {key} {command}")
        
    frames = []
    actions = []
    
    try:
        for i in range(module.episodes):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                raw_frame = state.screen_buffer
                
                processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
                action = game.get_last_action()
                
                frames.append(processed_frame)
                actions.append(action)
                game.advance_action()
                
        logger.info(f"Saving frames to {output_path}...")
        np.savez_compressed(output_path, frames=np.array(frames), actions=np.array(actions))
        
    except Exception as e:
        logger.error(f"Recording interrupted: {e}", exc_info=True)
    finally:
        game.close()