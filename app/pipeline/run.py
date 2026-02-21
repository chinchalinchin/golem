# Application Libraries
import logging
import time
from pathlib import Path

# External Libraries
import torch
import cv2
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# Application Libraries
from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, get_vizdoom_game, \
                        register_command, get_latest_parameters

logger = logging.getLogger(__name__)

@register_command("run")
def run(cfg: GolemConfig, module_name: str = "basic"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Loading Golem Brain on {device}...")

    active_profile = cfg.brain.mode
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    
    # 1. Base defaults
    cortical_depth = cfg.brain.cortical_depth
    working_memory = cfg.brain.working_memory
    n_actions = cfg.training.action_space_size 

    # 2. Discover architecture from archives
    archives = list(model_dir.glob("*.pth"))
    params = get_latest_parameters(archives)
    if params:
        cortical_depth, working_memory = params
        
    # 3. Discover action space and load state dict
    try:
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]
            
        model = DoomLiquidNet(
            n_actions=n_actions,
            cortical_depth=cortical_depth,
            working_memory=working_memory,
            sensors=cfg.brain.sensors
        ).to(device)
        
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Brain loaded successfully from {active_model_path}")
    except FileNotFoundError:
        logger.error(f"No model found at {active_model_path}. Please train first!")
        return

    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found.")
        return
    
    cfg_path = cfg.config[active_profile]
    scenario = cfg.modules[module_name].scenario

    game = get_vizdoom_game(cfg_path, scenario, cfg.brain.sensors)    
    game.init()

    logger.info("Golem is waking up...")
    
    # Pad action names if the runtime model has more actions than the default config
    action_labels = list(cfg.training.action_names)
    if len(action_labels) < n_actions:
        action_labels += [f"ACTION_{i}" for i in range(len(action_labels), n_actions)]
    
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        
        hx = None
        
        while not game.is_episode_finished():
            state = game.get_state()
            raw_frame = state.screen_buffer
            
            processed_frame = cv2.resize(raw_frame.transpose(1, 2, 0), (64, 64)) / 255.0
            x_vis_np = processed_frame
            
            if cfg.brain.sensors.depth and state.depth_buffer is not None:
                processed_depth = cv2.resize(state.depth_buffer, (64, 64)) / 255.0
                x_vis_np = np.concatenate((x_vis_np, np.expand_dims(processed_depth, axis=2)), axis=2)

            tensor_vis = torch.from_numpy(np.transpose(x_vis_np, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            
            tensor_aud = None
            if cfg.brain.sensors.audio and state.audio_buffer is not None:
                tensor_aud = torch.from_numpy(state.audio_buffer).float().unsqueeze(0).unsqueeze(0).to(device)
                
            with torch.no_grad():
                logits, hx = model(tensor_vis, x_aud=tensor_aud, hx=hx)
                probs = torch.sigmoid(logits)
            
            action_probs = probs.cpu().numpy()[0, 0]
            action = (action_probs > 0.5).astype(int).tolist()
            
            if sum(action) > 0 or (game.get_episode_time() % 35 == 0):
                active_str = " | ".join([f"{label}:{prob:.2f}" for label, prob in zip(action_labels, action_probs) if prob > 0.1])
                logger.info(f"T{game.get_episode_time()}: {active_str}")

            game.make_action(action)
            time.sleep(0.028)
        
        logger.info(f"Episode {i+1} Finished.")

    game.close()