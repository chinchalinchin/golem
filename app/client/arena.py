"""
Multiplayer Spectator Client.
Connects to the Host Arena to allow local human gameplay or observation.
"""
# Standard Libraries
import logging
import os
from pathlib import Path

# External Libraries
import numpy as np
import cv2
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# Application Libraries
from app.models.config import GolemConfig
from app.utils import resolve_path, get_vizdoom_scenario, get_unique_filename

logger = logging.getLogger(__name__)

def spectate(cfg: GolemConfig, module_name: str = "cig_arena"):
    logger.info("Initializing Local Spectator Client...")
    
    active_profile = cfg.brain.mode
    cfg_path = resolve_path(cfg.config[active_profile])
    scenario_path = get_vizdoom_scenario(cfg.modules[module_name].scenario)

    game = DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    
    # Enable Local Rendering
    game.set_window_visible(True)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    
    host_ip = os.getenv("HOST_IP", "127.0.0.1")
    agent_name = os.getenv("AGENT_NAME", "Human_Spectator")
    agent_color = os.getenv("AGENT_COLOR", "3")
    
    logger.info(f"Joining Host Server at {host_ip}...")
    game.add_game_args(f"-join {host_ip} -port 5029")
    game.add_game_args(f"+name {agent_name} +colorset {agent_color}")
    
    # SPECTATOR mode lets you use the keyboard natively
    game.set_mode(Mode.SPECTATOR) 
    game.init()

    logger.info("Welcome to the Arena. Press ESC in the Doom window to quit.")
    
    episodes = 5
    for i in range(episodes):
        while not game.is_episode_finished():
            # Advance the engine tick based on host sync
            game.advance_action()
        logger.info(f"Episode {i+1} Finished. Awaiting Host reset...")

    game.close()
    logger.info("Disconnected from Arena.")

def remote(cfg: GolemConfig, module_name: str = "cig_arena"):
    logger.info("Initializing Remote Recording Client...")
    
    active_profile = cfg.brain.mode
    cfg_path = resolve_path(cfg.config[active_profile])
    scenario_path = get_vizdoom_scenario(cfg.modules[module_name].scenario)

    game = DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    
    game.set_window_visible(True)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    
    host_ip = os.getenv("HOST_IP", "127.0.0.1")
    agent_name = os.getenv("AGENT_NAME", "Human_Trainer")
    agent_color = os.getenv("AGENT_COLOR", "2")
    
    game.add_game_args(f"-join {host_ip} -port 5029")
    game.add_game_args(f"+name {agent_name} +colorset {agent_color}")
    
    game.set_mode(Mode.SPECTATOR) 
    game.init()

    recorded_frames = []
    recorded_actions = []

    logger.info("Recording initiated. Fight!")
    
    episodes = 5
    for i in range(episodes):
        while not game.is_episode_finished():
            game.advance_action()
            
            state = game.get_state()
            if state is None:
                continue
                
            # Extract ground-truth action from what the human just pressed
            action = game.get_last_action()
            
            raw_frame = state.screen_buffer
            frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
            
            recorded_frames.append(frame)
            recorded_actions.append(action)

    game.close()
    
    if len(recorded_frames) > 0:
        frames_np = np.array(recorded_frames, dtype=np.float32)
        actions_np = np.array(recorded_actions, dtype=np.int8)
        
        target_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
        filename = get_unique_filename(target_dir, f"{cfg.data.prefix}{module_name}", "npz")
        
        np.savez_compressed(filename, frames=frames_np, actions=actions_np)
        logger.info(f"Multiplayer data saved: {filename} ({len(frames_np)} frames)")
    else:
        logger.warning("No data recorded.")