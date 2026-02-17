import logging
import cv2
import numpy as np
import vizdoom
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from app.config import GolemConfig
from app.utils import resolve_path, get_unique_filename

logger = logging.getLogger(__name__)

def get_vizdoom_resolution(res_str: str):
    """Maps config string to ViZDoom constant."""
    try:
        return getattr(ScreenResolution, res_str)
    except AttributeError:
        logger.warning(f"Resolution {res_str} not found, defaulting to RES_640X480")
        return ScreenResolution.RES_640X480

def get_package_scenario(name: str) -> str:
    """Finds built-in scenarios from the vizdoom package."""
    package_path = os.path.dirname(vizdoom.__file__)
    return os.path.join(package_path, "scenarios", name)

def record_data(cfg: GolemConfig):
    """Runs the game in spectator mode and records frames/actions."""
    
    # 1. Setup Paths
    cfg_path = resolve_path(cfg.vizdoom.config_path)
    output_dir = resolve_path(cfg.data.output_dir)
    output_path = get_unique_filename(output_dir, cfg.data.filename_prefix)
    
    logger.info(f"Initializing ViZDoom with config: {cfg_path}")
    
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_doom_scenario_path(get_package_scenario(cfg.vizdoom.scenario_name))
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(get_vizdoom_resolution(cfg.vizdoom.resolution))
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    
    # 2. Force Bindings (Fix for Spectator Mode)
    # We delay init until after settings, but before commands
    game.init()
    
    logger.debug("Injecting console command bindings...")
    game.send_game_command("bind a +moveleft")
    game.send_game_command("bind d +moveright")
    game.send_game_command("bind w +attack")
    game.send_game_command("bind space +attack")
    
    frames = []
    actions = []
    
    logger.info(f"Starting recording session for {cfg.vizdoom.episodes} episodes.")
    logger.info(f"Output target: {output_path}")
    
    try:
        for i in range(cfg.vizdoom.episodes):
            logger.info(f"Episode {i+1}/{cfg.vizdoom.episodes}")
            game.new_episode()
            
            while not game.is_episode_finished():
                state = game.get_state()
                
                # Extract
                raw_frame = state.screen_buffer
                
                # Transform: (Channels, Height, Width) -> (Height, Width, Channels) -> Resize
                processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64))
                processed_frame = processed_frame / 255.0
                
                # Extract Action
                action = game.get_last_action()
                
                frames.append(processed_frame)
                actions.append(action)
                
                game.advance_action()
                
        # Load (Save to disk)
        logger.info("Saving data to disk...")
        np.savez_compressed(output_path, frames=np.array(frames), actions=np.array(actions))
        logger.info(f"Successfully saved {len(frames)} frames to {output_path}")
        
    except Exception as e:
        logger.error(f"Recording interrupted: {e}", exc_info=True)
    finally:
        game.close()