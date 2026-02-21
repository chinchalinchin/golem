"""
Multiplayer Client Module.
Connects the LNN to the Host Arena Server.
"""
import torch
import cv2
import numpy as np
import logging
import os
from pathlib import Path
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, get_vizdoom_scenario

logger = logging.getLogger(__name__)

def run_client(cfg: GolemConfig, module_name: str = "cig_arena"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    active_profile = cfg.brain.mode
    model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    n_actions = cfg.training.action_space_size 
    
    logger.info(f"Loading {active_profile} Brain on {device}...")
    
    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory
    ).to(device) 
       
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Please train first!")
        return

    cfg_path = resolve_path(cfg.config[active_profile])
    scenario_path = get_vizdoom_scenario(cfg.modules[module_name].scenario)

    game = DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    
    # Headless Rendering for Docker
    game.set_window_visible(False)
    
    # Inject Docker Environment Variables
    host_ip = os.getenv("HOST_IP", "127.0.0.1")
    agent_name = os.getenv("AGENT_NAME", "Golem")
    agent_color = os.getenv("AGENT_COLOR", "1")
    
    logger.info(f"Attempting to join Host Server at {host_ip}...")
    game.add_game_args(f"-join {host_ip}")
    game.add_game_args(f"+name {agent_name} +colorset {agent_color}")
    
    game.set_mode(Mode.PLAYER) 
    game.init()

    logger.info(f"{agent_name} has entered the Arena!")
    
    action_labels = cfg.training.action_names
    
    episodes = 5
    for i in range(episodes):
        # The Host strictly dictates episode resets. 
        hx = None
        
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                # Occurs during early host sync waits
                game.advance_action()
                continue
                
            raw_frame = state.screen_buffer
            frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
            tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, hx = model(tensor, hx)
                probs = torch.sigmoid(logits)
            
            action_probs = probs.cpu().numpy()[0, 0]
            action = (action_probs > 0.5).astype(int).tolist()
            
            if sum(action) > 0 or (game.get_episode_time() % 35 == 0):
                active_str = " | ".join([f"{label}:{prob:.2f}" for label, prob in zip(action_labels, action_probs) if prob > 0.1])
                logger.info(f"[{agent_name}] T{game.get_episode_time()}: {active_str}")

            game.make_action(action)
        
        logger.info(f"Episode {i+1} Finished. Waiting for Host to restart...")

    game.close()