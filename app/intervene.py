"""
DAgger Module: Intervention and Dataset Aggregation.
Allows the human expert to take over control from the LNN to correct mistakes.
"""
import torch
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from app.config import GolemConfig
from app.brain import DoomLiquidNet
from app.utils import resolve_path, get_unique_filename, get_vizdoom_scenario

try:
    from pynput import keyboard
except ImportError:
    raise ImportError("pynput is required for DAgger. Run: pip install pynput")

logger = logging.getLogger(__name__)

class InterventionController:
    """Background listener to capture raw OS keyboard state during overrides."""
    def __init__(self, action_names):
        self.action_names = action_names
        self.intervening = False
        self.keys_pressed = set()
        
        # Hardcoded map for the DAgger override
        self.key_map = {
            'w': 'MOVE_FORWARD',
            's': 'MOVE_BACKWARD',
            'a': 'MOVE_LEFT',
            'd': 'MOVE_RIGHT',
            keyboard.Key.left: 'TURN_LEFT',
            keyboard.Key.right: 'TURN_RIGHT',
            keyboard.Key.space: 'ATTACK',
            'e': 'USE',
            'q': 'SELECT_NEXT_WEAPON'
        }
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _normalize_key(self, key):
        return key.char.lower() if hasattr(key, 'char') and key.char else key

    def on_press(self, key):
        if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            self.intervening = True
        self.keys_pressed.add(self._normalize_key(key))

    def on_release(self, key):
        if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            self.intervening = False
            
        norm_key = self._normalize_key(key)
        if norm_key in self.keys_pressed:
            self.keys_pressed.remove(norm_key)

    def get_action_vector(self):
        vector = [0] * len(self.action_names)
        for k in self.keys_pressed:
            action_name = self.key_map.get(k)
            if action_name and action_name in self.action_names:
                idx = self.action_names.index(action_name)
                vector[idx] = 1
        return vector

def intervene_agent(cfg: GolemConfig, module_name: str = "combat"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Loading Golem Brain on {device} for DAgger Intervention...")

    # Load Model
    model_path = resolve_path(cfg.training.model_save_path)
    n_actions = cfg.training.action_space_size 
    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory
    ).to(device) 
       
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error(f"No model found. Train first!")
        return

    # Setup Environment
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

    action_labels = cfg.training.action_names
    controller = InterventionController(action_labels)
    
    # Memory Buffer
    recovery_frames = []
    recovery_actions = []

    logger.info("======================================================")
    logger.info("DAgger Mode Active. Golem is running autonomously.")
    logger.info("HOLD [LEFT SHIFT] to pause the LNN and take manual control.")
    logger.info("Use WASD, Arrows, Space, and Q to steer.")
    logger.info("Release [LEFT SHIFT] to save the recovery memory and resume.")
    logger.info("======================================================")
    
    episodes = 5
    for i in range(episodes):
        game.new_episode()
        hx = None
        
        while not game.is_episode_finished():
            state = game.get_state()
            raw_frame = state.screen_buffer
            
            # Process Frame for LNN (and potentially saving)
            processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
            tensor = torch.from_numpy(np.transpose(processed_frame, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # 1. Forward Pass (Keep the ODE state updating even during override)
            with torch.no_grad():
                logits, hx = model(tensor, hx)
                probs = torch.sigmoid(logits)
            
            # 2. Control Routing
            if controller.intervening:
                # HUMAN OVERRIDE
                action = controller.get_action_vector()
                
                # Record to buffer
                recovery_frames.append(processed_frame)
                recovery_actions.append(action)
                
                if game.get_episode_time() % 35 == 0:
                    logger.warning(f"OVERRIDE ACTIVE | Recording frame {len(recovery_frames)}...")
            else:
                # LNN AUTONOMY
                action_probs = probs.cpu().numpy()[0, 0]
                action = (action_probs > 0.5).astype(int).tolist()
                
                # If we just finished an intervention, flush the buffer to disk
                if len(recovery_frames) > 0:
                    output_dir = resolve_path(cfg.data.output_dir)
                    prefix = f"{cfg.data.filename_prefix}_{active_profile}_{module_name}_recovery"
                    output_path = get_unique_filename(output_dir, prefix)
                    
                    logger.info(f"Saving {len(recovery_frames)} recovery frames to {output_path}...")
                    np.savez_compressed(output_path, frames=np.array(recovery_frames), actions=np.array(recovery_actions))
                    
                    # Clear buffer
                    recovery_frames = []
                    recovery_actions = []

            game.make_action(action)
            time.sleep(0.028)
            
        logger.info(f"Episode {i+1} Finished.")

    game.close()