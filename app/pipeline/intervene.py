"""
DAgger Module: Intervention and Dataset Aggregation.
Allows the human expert to take over control from the LNN to correct mistakes.
"""
# Standard Libraries
import logging
import time
from pathlib import Path

# External Libraries
import torch
import torchaudio
import cv2
import numpy as np

# Application Libraries
from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, get_unique_filename, \
                         get_vizdoom_game, register_command

logger = logging.getLogger(__name__)

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
        sensors=cfg.brain.sensors
    ).to(device)
       
    mel_transform = None
    amp_to_db = None
    if cfg.brain.sensors.audio:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.brain.dsp.sample_rate,
            n_fft=cfg.brain.dsp.n_fft,
            hop_length=cfg.brain.dsp.hop_length,
            n_mels=cfg.brain.dsp.n_mels
        ).to(device)
        amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

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
    scenario = cfg.modules[module_name].scenario

    game = get_vizdoom_game(cfg_path, scenario, cfg.brain.sensors)
    game.init()

    # Get active bindings for the enabled brain mode
    active_bindings = cfg.keybindings.get(active_profile, {})
    action_labels = cfg.training.action_names
    
    # Pass bindings to dynamically map keys
    controller = InterventionController(action_labels, active_bindings)
    
    # Memory Buffer
    recovery_frames = []
    recovery_depths = []
    recovery_audios = []
    recovery_actions = []
    recovery_thermals = []
    
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
            
            processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
            x_vis_np = processed_frame
            
            processed_depth = None
            if cfg.brain.sensors.depth and state.depth_buffer is not None:
                processed_depth = cv2.resize(state.depth_buffer, (64, 64)) / 255.0
                x_vis_np = np.concatenate((x_vis_np, np.expand_dims(processed_depth, axis=2)), axis=2)
                
            raw_audio = None
            tensor_aud = None
            if cfg.brain.sensors.audio and state.audio_buffer is not None:
                raw_audio = state.audio_buffer
                mean = np.mean(raw_audio, axis=-1, keepdims=True)
                std = np.std(raw_audio, axis=-1, keepdims=True) + 1e-8
                norm_audio = (raw_audio - mean) / std
                
                # In intervene.py, if you are appending recovery frames, append `norm_audio` 
                # (NOT the spectrogram, because dataset.py expects normalized raw audio arrays)
                
                # Transform to Spectrogram strictly for live tensor inference
                tensor_aud = torch.from_numpy(norm_audio).float().unsqueeze(0).unsqueeze(0).to(device)
                tensor_aud = mel_transform(tensor_aud)
                tensor_aud = amp_to_db(tensor_aud)

            processed_thermal = None
            if cfg.brain.sensors.thermal and state.labels_buffer is not None:
                binary_mask = (state.labels_buffer > 0).astype(np.float32)
                processed_thermal = cv2.resize(binary_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
 
            tensor_vis = torch.from_numpy(np.transpose(x_vis_np, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            tensor_aud = torch.from_numpy(raw_audio).float().unsqueeze(0).unsqueeze(0).to(device) if raw_audio is not None else None
            tensor_thm = torch.from_numpy(processed_thermal).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) if processed_thermal is not None else None

            with torch.no_grad():
                logits, hx = model(tensor_vis, x_aud=tensor_aud, x_thm=tensor_thm, hx=hx)                
                probs = torch.sigmoid(logits)
            
            if controller.intervening:
                action = controller.get_action_vector()
                recovery_frames.append(processed_frame)
                
                if processed_depth is not None:
                    recovery_depths.append(processed_depth)
                if raw_audio is not None:
                    recovery_audios.append(norm_audio) 
                if processed_thermal is not None:
                    recovery_thermals.append(processed_thermal)
                                              
                recovery_actions.append(action)
                
                if game.get_episode_time() % 35 == 0:
                    logger.warning(f"OVERRIDE ACTIVE | Recording frame {len(recovery_frames)}...")
            else:
                action_probs = probs.cpu().numpy()[0, 0]
                action = (action_probs > 0.5).astype(int).tolist()
                
                if len(recovery_frames) > 0:
                    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
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
                    
                    recovery_frames, recovery_depths, recovery_audios, recovery_thermals, recovery_actions = [], [], [], [], []
            game.make_action(action)
            time.sleep(0.028)
            
        logger.info(f"Episode {i+1} Finished.")

    game.close()