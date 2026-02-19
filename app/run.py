import torch
import cv2
import numpy as np
import logging
import time
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from app.config import GolemConfig
from app.brain import DoomLiquidNet
from app.utils import resolve_path, get_vizdoom_scenario

logger = logging.getLogger(__name__)

def run_agent(cfg: GolemConfig, module_name: str = "basic"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Loading Golem Brain on {device}...")

    model_path = resolve_path(cfg.training.model_save_path)
    n_actions = cfg.training.action_space_size 
    model = DoomLiquidNet(n_actions=n_actions).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Brain loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Please train first!")
        return

    active_profile = cfg.training.config
    cfg_path = resolve_path(cfg.config[active_profile])

    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found.")
        return
    scenario_path = get_vizdoom_scenario(cfg.modules[module_name].scenario)

    game = DoomGame()
    game.load_config(resolve_path(cfg_path))    
    game.set_doom_scenario_path(scenario_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER) 
    game.init()

    logger.info("Golem is waking up...")
    
    action_labels = cfg.training.action_names
    
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        
        # Reset Short-term memory (Hidden State) at start of episode
        hx = None
        
        while not game.is_episode_finished():
            state = game.get_state()
            raw_frame = state.screen_buffer
            
            frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            
            tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # FIX: Pass and receive hx
                logits, hx = model(tensor, hx)
                probs = torch.sigmoid(logits)
            
            action_probs = probs.cpu().numpy()[0, 0]
            action = (action_probs > 0.5).astype(int).tolist()
            
            # Neural Monitor: Log thoughts if something is happening
            # Only log every 10 frames to avoid spam, or if an action is taken
            if sum(action) > 0 or (game.get_episode_time() % 35 == 0):
                active_str = " | ".join([f"{label}:{prob:.2f}" for label, prob in zip(action_labels, action_probs) if prob > 0.1])
                logger.info(f"T{game.get_episode_time()}: {active_str}")

            game.make_action(action)
            time.sleep(0.028)
        
        logger.info(f"Episode {i+1} Finished.")

    game.close()