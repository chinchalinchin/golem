# Standard Libraries
import logging
from pathlib import Path

# External Libraries
import numpy as np
import vizdoom

# Application Libraries
from app.models.config import GolemConfig
from app.utils import resolve_path, get_unique_filename, \
                        get_vizdoom_game, register_command, \
                        SensoryExtractor

logger = logging.getLogger(__name__)

@register_command("record")
def record(cfg: GolemConfig, module_name: str = "basic"):
    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found. Available: {list(cfg.modules.keys())}")
        return

    module = cfg.modules[module_name]
    active_profile = cfg.brain.mode
    
    if active_profile not in cfg.config:
        logger.error(f"Profile '{active_profile}' not found in config mapping.")
        return
        
    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    prefix_clean = cfg.data.prefix.rstrip('_')
    file_prefix = f"{prefix_clean}_{module_name}"
    output_path = get_unique_filename(output_dir, file_prefix, "npz")

    cfg_path = cfg.config[active_profile]
    scenario = module.scenario
    map_name = module.map

    logger.info(f"--- Recording Module: {module_name} ---")
    
    # Explicitly set SPECTATOR mode to allow manual human input via keyboard
    game = get_vizdoom_game(cfg_path, scenario, cfg.brain.sensors, mode=vizdoom.Mode.SPECTATOR, map_name=map_name)
    game.init()
    
    active_bindings = cfg.keybindings.get(active_profile, {})
    logger.info(f"Injecting Keybindings: {active_bindings}")
    
    for key, command in active_bindings.items():
        game.send_game_command(f"bind {key} {command}")
        
    frames = []
    depths = []
    audios = []
    thermals = []
    actions = []
    
    try:
        for _ in range(module.episodes):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                action = game.get_last_action()
                
                # Centralized Extraction
                extracted = SensoryExtractor.get_numpy_state(state, cfg.brain.sensors)
                
                if 'visual' in extracted: frames.append(extracted['visual'])
                if 'depth' in extracted: depths.append(extracted['depth'])
                if 'audio' in extracted: audios.append(extracted['audio'])
                if 'thermal' in extracted: thermals.append(extracted['thermal'])
                
                actions.append(action)
                game.advance_action()
                
        logger.info(f"Saving frames to {output_path}...")
        save_dict = {'frames': np.array(frames), 'actions': np.array(actions)}
        
        if cfg.brain.sensors.depth:
            save_dict['depths'] = np.array(depths)
        if cfg.brain.sensors.audio:
            save_dict['audios'] = np.array(audios)
        if cfg.brain.sensors.thermal:
            save_dict['thermals'] = np.array(thermals) 
        
        np.savez_compressed(output_path, **save_dict)
        
    except Exception as e:
        logger.error(f"Recording interrupted: {e}", exc_info=True)
    finally:
        game.close()