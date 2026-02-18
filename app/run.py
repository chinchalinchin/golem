"""
Inference Module: The Body.

This module loads a trained Golem Brain (PyTorch model) and connects it 
to a live instance of ViZDoom. It runs in PLAYER mode (ASYNC), allowing 
the neural network to drive the game loop by predicting actions from 
pixel data in real-time.
"""
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

def run_agent(cfg: GolemConfig):
    """
    Main inference loop.
    
    1. Loads the model weights from disk.
    2. Initializes ViZDoom in PLAYER mode.
    3. Runs the game loop: Frame -> Tensor -> Model -> Action -> Game.
    """
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Loading Golem Brain on {device}...")

    # 2. Load the Trained Brain
    model_path = resolve_path(cfg.training.model_save_path)
    model = DoomLiquidNet(n_actions=3).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Brain loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Please train first!")
        return

    # 3. Initialize Game
    game = DoomGame()
    game.load_config(resolve_path(cfg.vizdoom.config_path))
    game.set_doom_scenario_path(get_vizdoom_scenario(cfg.vizdoom.scenario_name))
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER) 
    game.init()

    logger.info("Golem is waking up...")
    
    # Run loop
    episodes = 5
    for i in range(episodes):
        game.new_episode()
        
        while not game.is_episode_finished():
            state = game.get_state()
            raw_frame = state.screen_buffer
            
            # Preprocess (Must match record.py logic!)
            frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1)) # (C, H, W) for PyTorch
            
            tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)
                
            action_probs = probs.cpu().numpy()[0, 0]
            
            # Thresholding Logic
            action = [0, 0, 0] # [Left, Right, Attack]
            if action_probs[0] > 0.5: action[0] = 1 
            if action_probs[1] > 0.5: action[1] = 1 
            if action_probs[2] > 0.5: action[2] = 1 
            
            # Conflicting movement suppression
            if action[0] and action[1]:
                if action_probs[0] > action_probs[1]: action[1] = 0
                else: action[0] = 0

            game.make_action(action)
            
            # Frame rate cap (approx 35 fps)
            time.sleep(0.028)
        
        logger.info(f"Episode {i+1} Finished.")

    game.close()