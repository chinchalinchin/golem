"""
DAgger Module: Intervention and Dataset Aggregation.
Allows the human expert to take over control from the LNN to correct mistakes.
"""
# Standard Libraries
import logging
import time
from pathlib import Path
from collections import deque

# External Libraries
import torch
import numpy as np

# Application Libraries
from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils.conf import resolve_path, get_unique_filename, register_command
from app.utils.model import SensoryExtractor
from app.utils.doom import get_game


logger = logging.getLogger(__name__)

# Module Level Constants
# 1. Translation Dictionary: Doom Command -> ViZDoom Action
ACTION_MAP = {
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
PYNPUT_MAP = {
    'leftarrow': 'left',
    'rightarrow': 'right',
    'space': 'space'
}


class InterventionController:
    """
    Background listener to capture raw OS keyboard state during overrides.
    
    Pynput is lazily loaded for headless environments.
    """
    def __init__(self, action_names, active_bindings):
        self.action_names = action_names
        self.intervening = False
        self.keys_pressed = set()
    
        
        # Dynamically build the key map
        self.key_map = {}
        for yaml_key, doom_cmd in active_bindings.items():
            norm_key = PYNPUT_MAP.get(yaml_key, str(yaml_key).lower())
            action_name = ACTION_MAP.get(doom_cmd)
            
            if action_name:
                self.key_map[norm_key] = action_name
                
        logger.debug(f"Intervention Controller mapped keys: {self.key_map}")
        
        # 1. Lazy import inside the constructor
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError("pynput requires a display/X11 server. Cannot run in headless mode.") from e

        # 2. Initialize the listener using the locally scoped keyboard module
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
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
        from pynput import keyboard
        if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            self.intervening = True
        self.keys_pressed.add(self._normalize_key(key))

    def on_release(self, key):
        from pynput import keyboard
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

@register_command("intervene")
def intervene(cfg: GolemConfig, module_name: str = "combat"):
    try:
        from pynput import keyboard
    except ImportError:
        logger.error("pynput is required for DAgger. Run: pip install pynput")
        return
    
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
        working_memory=cfg.brain.working_memory,
        sensors=cfg.brain.sensors,
        dsp_config=cfg.brain.dsp
    ).to(device)
       
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error(f"No model found at {model_path}. Train first!")
        return

    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found.")
        return

    # Setup Environment using brain.mode
    active_profile = cfg.brain.mode
    cfg_path = cfg.config[active_profile]
    module = cfg.modules[module_name]
    scenario = module.scenario
    map_name = module.map

    game = get_game(cfg_path, scenario, cfg.brain.sensors, map_name=map_name)
    game.init()

    # Get active bindings for the enabled brain mode
    active_bindings = cfg.keybindings.get(active_profile, {})
    action_labels = cfg.training.action_names
    
    # Pass bindings to dynamically map keys
    controller = InterventionController(action_labels, active_bindings)
    
    # Rolling Context Buffers
    auto_frames = deque(maxlen=cfg.training.sequence_length)
    auto_depths = deque(maxlen=cfg.training.sequence_length)
    auto_audios = deque(maxlen=cfg.training.sequence_length)
    auto_thermals = deque(maxlen=cfg.training.sequence_length)
    auto_actions = deque(maxlen=cfg.training.sequence_length)

    # Memory Buffer
    recovery_frames, recovery_depths, recovery_audios, recovery_actions, recovery_thermals \
            = [], [], [], [], []

    logger.info("======================================================")
    logger.info("DAgger Mode Active. Golem is running autonomously.")
    logger.info("HOLD [LEFT SHIFT] to pause the LNN and take manual control.")
    logger.info("Use WASD, Arrows, Space, and Q to steer.")
    logger.info("Release [LEFT SHIFT] to save the recovery memory and resume.")
    logger.info("======================================================")
    
    # TODO: make this configurable or pass it in through the CLI
    episodes = 5
    was_intervening = False
    for i in range(episodes):
        game.new_episode()
        hx = None
        
        while not game.is_episode_finished():
            # Physiological Reset (Death)
            if game.is_player_dead():
                hx = None
                
            state = game.get_state()
            if state is None:
                game.advance_action()
                continue

            # Centralized Extraction & Tensor formatting
            extracted = SensoryExtractor.get_numpy_state(state, cfg.brain.sensors)
            tensors = SensoryExtractor.to_tensors(extracted, device)

            with torch.no_grad():
                logits, hx = model(
                    tensors.get('visual'), 
                    x_aud=tensors.get('audio'), 
                    x_thm=tensors.get('thermal'), 
                    hx=hx
                )                
                probs = torch.sigmoid(logits)
            
            if controller.intervening:
                # 1. Fetch the human's corrective action FIRST
                action = controller.get_action_vector()

                if not was_intervening:
                    # 2. Flush the autonomous rolling context into the recovery buffers
                    recovery_frames.extend(auto_frames)
                    if cfg.brain.sensors.depth: recovery_depths.extend(auto_depths)
                    if cfg.brain.sensors.audio: recovery_audios.extend(auto_audios)
                    if cfg.brain.sensors.thermal: recovery_thermals.extend(auto_thermals)
                    
                    # 3. RETROACTIVE CORRECTION: Apply the human action to the historical frames
                    recovery_actions.extend([action] * len(auto_frames))
                    
                    was_intervening = True
                    
                # Append current active intervention arrays 
                if 'visual' in extracted: recovery_frames.append(extracted['visual'])
                if 'depth' in extracted: recovery_depths.append(extracted['depth'])
                if 'audio' in extracted: recovery_audios.append(extracted['audio'])
                if 'thermal' in extracted: recovery_thermals.append(extracted['thermal'])
                                          
                recovery_actions.append(action)
                
                if game.get_episode_time() % 35 == 0:
                    logger.warning(f"OVERRIDE ACTIVE | Recording frame {len(recovery_frames)}...")
            else:
                was_intervening = False
                action_probs = probs.cpu().numpy()[0, 0]
                action = (action_probs > 0.5).astype(int).tolist()
                
                # POPULATE THE AUTONOMOUS ROLLING BUFFERS
                if 'visual' in extracted: auto_frames.append(extracted['visual'])
                if 'depth' in extracted: auto_depths.append(extracted['depth'])
                if 'audio' in extracted: auto_audios.append(extracted['audio'])
                if 'thermal' in extracted: auto_thermals.append(extracted['thermal'])
                auto_actions.append(action)
                
                # Save block (Triggers when the user releases the intervention key)
                if len(recovery_frames) > 0:
                    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "recovery"
                    prefix_clean = cfg.data.prefix.rstrip('_')
                    file_prefix = f"{prefix_clean}_{module_name}_recovery"
                    
                    output_path = get_unique_filename(output_dir, file_prefix, "npz")
                    logger.info(f"Saving {len(recovery_frames)} recovery frames to {output_path}...")
                    
                    save_dict = {'frames': np.array(recovery_frames), 'actions': np.array(recovery_actions)}
                    if len(recovery_depths) > 0:
                        save_dict['depths'] = np.array(recovery_depths)
                    if len(recovery_audios) > 0:
                        save_dict['audios'] = np.array(recovery_audios)
                    if len(recovery_thermals) > 0:
                        save_dict['thermals'] = np.array(recovery_thermals)

                    np.savez_compressed(output_path, **save_dict)
                    
                    # Clear memory buffers
                    recovery_frames, recovery_depths, recovery_audios, recovery_thermals, recovery_actions = [], [], [], [], []
            
            game.make_action(action)
            time.sleep(0.028)
            
        logger.info(f"Episode {i+1} Finished.")

    game.close()