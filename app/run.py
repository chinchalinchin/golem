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
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def run_agent(cfg: GolemConfig):
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
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode (freezes Dropout, etc.)
        logger.info(f"Brain loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Please train first!")
        return

    # 3. Initialize Game (Player Mode)
    game = DoomGame()
    game.load_config(resolve_path(cfg.vizdoom.config_path))
    game.set_doom_scenario_path(resolve_path(f"conf/{cfg.vizdoom.scenario_name}") if "conf" not in cfg.vizdoom.scenario_name else resolve_path(cfg.vizdoom.scenario_name))
    
    # We need to find the WAD. Using the built-in one for now as in record.py
    import vizdoom as vz
    import os
    package_path = os.path.dirname(vz.__file__)
    game.set_doom_scenario_path(os.path.join(package_path, "scenarios", cfg.vizdoom.scenario_name))

    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    
    # CRITICAL: Switch to ASYNC_PLAYER so the bot can think while the game runs
    # OR SPECTATOR if you just want to watch it fail, but PLAYER is standard for bots
    game.set_mode(Mode.PLAYER) 
    game.init()

    logger.info("Golem is waking up...")
    
    episodes = 5
    for i in range(episodes):
        game.new_episode()
        
        while not game.is_episode_finished():
            # Get State
            state = game.get_state()
            raw_frame = state.screen_buffer
            
            # Preprocess (Same as Training!)
            # 1. Resize -> 64x64
            # 2. Normalize -> 0-1
            # 3. Transpose -> (Channels, Height, Width)
            frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            
            # Convert to Tensor
            # Add Batch Dimension: (1, Channels, Height, Width) -> (1, Time=1, C, H, W)
            # The CfC expects a time dimension. We feed it 1 frame at a time.
            tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(tensor)
                # Apply Sigmoid to get probabilities (0.0 to 1.0)
                probs = torch.sigmoid(logits)
                
            # Decision Boundary (Threshold > 0.5)
            # Output: [Left, Right, Attack]
            action_probs = probs.cpu().numpy()[0, 0] # Squeeze batch and time
            
            # Convert to VizDoom Action List [0.0, 1.0, 0.0]
            # We map: 0->Left, 1->Right, 2->Attack
            # Note: VizDoom usually expects ints or booleans for buttons
            action = [0, 0, 0]
            
            # Simple Threshold Logic
            if action_probs[0] > 0.5: action[0] = 1 # Left
            if action_probs[1] > 0.5: action[1] = 1 # Right
            if action_probs[2] > 0.5: action[2] = 1 # Attack
            
            # If confusingly both left and right, prioritize the stronger signal
            if action[0] and action[1]:
                if action_probs[0] > action_probs[1]: action[1] = 0
                else: action[0] = 0

            # Execute
            game.make_action(action)
            
            # Optional: Sleep to match roughly 30fps if it runs too fast
            time.sleep(0.028)
        
        logger.info(f"Episode {i+1} Finished.")

    game.close()