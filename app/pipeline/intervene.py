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

from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, get_unique_filename, get_vizdoom_scenario

try:
    from pynput import keyboard
except ImportError:
    raise ImportError("pynput is required for DAgger. Run: pip install pynput")

logger = logging.getLogger(__name__)

class InterventionController:
    """Background listener to capture raw OS keyboard state during overrides."""
    def __init__(self, action_names, active_bindings):
        self.action_names = action_names
        self.intervening = False
        self.keys_pressed = set()
        
        # 1. Translation Dictionary: Doom Command -> ViZDoom Action
        command_to_action = {
            "+forward": "MOVE_FORWARD",
            "+back": "MOVE_BACKWARD",
            "+moveleft": "MOVE_LEFT",
            "+moveright": "MOVE_RIGHT",
            "+left": "TURN_LEFT",
            "+right": "TURN_RIGHT",
            "+attack": "ATTACK",
            "+use": "USE",
            "weapnext": "SELECT_NEXT_WEAPON",
            '"slot 2"': "SELECT_WEAPON2",
            '"slot 3"': "SELECT_WEAPON3"
        }
        
        # 2. Translation Dictionary: YAML String -> pynput Key Name
        yaml_to_pynput = {
            'leftarrow': 'left',
            'rightarrow': 'right',
            'space': 'space'
        }
        
        # 3. Dynamically build the key map
        self.key_map = {}
        for yaml_key, doom_cmd in active_bindings.items():
            norm_key = yaml_to_pynput.get(yaml_key, str(yaml_key).lower())
            action_name = command_to_action.get(doom_cmd)
            
            if action_name:
                self.key_map[norm_key] = action_name
                
        logger.debug(f"Intervention Controller mapped keys: {self.key_map}")
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _normalize_key(self, key):
        """Converts pynput key events into normalized string representations."""
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        elif hasattr(key, 'name') and key.name:
            # Handles special keys like keyboard.Key.space -> 'space'
            return key.name.lower()
        return str(key)

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
    
# -----------------------------------------------------------------

def intervene_agent(cfg: GolemConfig, module_name: str = "combat"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Loading Golem Brain on {device} for DAgger Intervention...")

    # Load Model from standard active model location
    active_profile = cfg.brain.mode
    model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    n_actions = cfg.training.action_space_size 
    
    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory
    ).to(device) 
       
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Train first!")
        return

    # Setup Environment using brain.mode
    active_profile = cfg.brain.mode
    cfg_path = resolve_path(cfg.config[active_profile])

    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found.")
        return
        
    scenario_path = get_vizdoom_scenario(cfg.modules[module_name].scenario)

    game = DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER) 
    game.init()

    # Get active bindings for the enabled brain mode
    active_bindings = cfg.keybindings.get(active_profile, {})
    action_labels = cfg.training.action_names
    
    # Pass bindings to dynamically map keys
    controller = InterventionController(action_labels, active_bindings)
    
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
                    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
                    prefix_clean = cfg.data.prefix.rstrip('_')
                    file_prefix = f"{prefix_clean}_{module_name}_recovery"
                    
                    output_path = get_unique_filename(output_dir, file_prefix, "npz")
                    
                    logger.info(f"Saving {len(recovery_frames)} recovery frames to {output_path}...")
                    np.savez_compressed(output_path, frames=np.array(recovery_frames), actions=np.array(recovery_actions))
                    
                    # Clear buffer
                    recovery_frames = []
                    recovery_actions = []

            game.make_action(action)
            time.sleep(0.028)
            
        logger.info(f"Episode {i+1} Finished.")

    game.close()