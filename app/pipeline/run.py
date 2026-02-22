# Application Libraries
import logging
import time
from pathlib import Path

# External Libraries
import torch

# Application Libraries
from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils.conf import resolve_path, register_command
from app.utils.model import apply_latest_parameters, SensoryExtractor
from app.utils.doom import get_game

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
    
    n_actions = cfg.training.action_space_size 

    # 1. Discover architecture from archives
    archives = list(model_dir.glob("*.pth"))
    apply_latest_parameters(cfg, archives)
        
    # 3. Discover action space and load state dict
    try:
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]
            
        model = DoomLiquidNet(
            n_actions=n_actions,
            cortical_depth=cfg.brain.cortical_depth,
            working_memory=cfg.brain.working_memory,
            sensors=cfg.brain.sensors,
            dsp_config=cfg.brain.dsp
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
    module = cfg.modules[module_name]
    scenario = module.scenario
    map_name = module.map

    game = get_game(cfg_path, scenario, cfg.brain.sensors, map_name=map_name)   
    game.init()

    logger.info("Golem is waking up...")
    
    # Pad action names if the runtime model has more actions than the default config
    action_labels = list(cfg.training.action_names)
    if len(action_labels) < n_actions:
        action_labels += [f"ACTION_{i}" for i in range(len(action_labels), n_actions)]

    # TODO: make this configurable, or pass it in through CLI.
    episodes = 10
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
            
            action_probs = probs.cpu().numpy()[0, 0]
            action = (action_probs > 0.5).astype(int).tolist()
            
            if sum(action) > 0 or (game.get_episode_time() % 35 == 0):
                active_str = " | ".join([f"{label}:{prob:.2f}" for label, prob in zip(action_labels, action_probs) if prob > 0.1])
                logger.info(f"T{game.get_episode_time()}: {active_str}")

            game.make_action(action)
            time.sleep(0.028)
        
        logger.info(f"Episode {i+1} Finished.")

    game.close()