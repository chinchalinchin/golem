"""
ETL Module: Recording.
Handles capturing gameplay for specific modules.
"""
import logging
import cv2
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from app.config import GolemConfig
from app.utils import resolve_path, get_unique_filename, get_vizdoom_scenario

logger = logging.getLogger(__name__)

def record_data(cfg: GolemConfig, module_name: str = "basic"):
    """
    Records gameplay for a specific module.
    
    Args:
        cfg: The global app configuration.
        module_name: Key in the 'modules' dictionary to record.
    """
    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found in configuration. Available: {list(cfg.modules.keys())}")
        return

    module = cfg.modules[module_name]
    
    # 1. Setup Paths
    # We use the module name in the filename: data/doom_training_basic_1.npz
    output_dir = resolve_path(cfg.data.output_dir)
    prefix = f"{cfg.data.filename_prefix}_{module_name}"
    output_path = get_unique_filename(output_dir, prefix)
    
    cfg_path = resolve_path(module.config)
    scenario_path = get_vizdoom_scenario(module.scenario)

    logger.info(f"--- Recording Module: {module_name} ---")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Scenario: {scenario_path}")
    
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_doom_scenario_path(scenario_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    
    game.init()
    
    # Inject Bindings for the FULL Action Space (8 buttons)
    # This ensures consistency even if the module doesn't strictly require all of them.
    logger.debug("Injecting superset bindings...")
    game.send_game_command("bind w +forward")
    game.send_game_command("bind s +back")
    game.send_game_command("bind a +moveleft")
    game.send_game_command("bind d +moveright")
    game.send_game_command("bind q +left")  # Turn Left
    game.send_game_command("bind e +right") # Turn Right
    game.send_game_command("bind space +attack")
    game.send_game_command("bind f +use")

    frames = []
    actions = []
    
    logger.info(f"Starting recording session for {module.episodes} episodes.")
    logger.info("Controls: W/S (Mov), A/D (Strafe), Q/E (Turn), Space (Fire), F (Use)")
    
    try:
        for i in range(module.episodes):
            logger.info(f"Episode {i+1}/{module.episodes}")
            game.new_episode()
            
            while not game.is_episode_finished():
                state = game.get_state()
                raw_frame = state.screen_buffer
                
                # Transform
                processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64))
                processed_frame = processed_frame / 255.0
                
                action = game.get_last_action()
                
                frames.append(processed_frame)
                actions.append(action)
                
                game.advance_action()
                
        logger.info(f"Saving {len(frames)} frames to {output_path}...")
        np.savez_compressed(output_path, frames=np.array(frames), actions=np.array(actions))
        
    except Exception as e:
        logger.error(f"Recording interrupted: {e}", exc_info=True)
    finally:
        game.close()