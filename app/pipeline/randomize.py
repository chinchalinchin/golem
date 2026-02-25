"""
Randomize Module: Procedural Generation Data Pipeline.
"""
# Standard Libraries
import logging
import random
from pathlib import Path

# External Libraries
import numpy as np
import vizdoom

# Application Libraries
from app.models.config import GolemConfig, SensorsConfig
from app.utils.conf import resolve_path, get_unique_filename, register_command
from app.utils.doom import get_game, ObligeGenerator
from app.utils.model import SensoryExtractor

logger = logging.getLogger(__name__)

@register_command("randomize")
def randomize(cfg: GolemConfig):
    try:
        from pynput import keyboard
    except ImportError:
        logger.error("pynput is required for manual termination. Run: pip install pynput")
        return

    active_profile = cfg.brain.mode
    if active_profile not in cfg.config:
        logger.error(f"Profile '{active_profile}' not found in config mapping.")
        return

    output_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_clean = cfg.data.prefix.rstrip('_')
    file_prefix = f"{prefix_clean}_random"
    
    cfg_path = cfg.config[active_profile]
    
    all_sensors = SensorsConfig(visual=True, depth=True, audio=True, thermal=True)
    active_bindings = cfg.keybindings.get(active_profile, {})

    iterations = cfg.randomizer.iterations
    duration_secs = cfg.randomizer.duration
    max_tics = duration_secs * 35

    logger.info(f"--- Randomize Pipeline: {iterations} iterations, {duration_secs}s per map ---")
    logger.info("Press [TAB] at any time to kill the current map early and skip to the next.")
    logger.info("Press [ESC] at any time to abort the entire pipeline.")

    # Instantiate the existing generator wrapper
    generator = ObligeGenerator(cfg.randomizer)

    stop_recording = False
    abort_all = False

    def on_press(key):
        nonlocal stop_recording, abort_all
        if key == keyboard.Key.tab:
            stop_recording = True
        elif key == keyboard.Key.esc:
            logger.warning("ESC pressed! Aborting entire pipeline.")
            stop_recording = True
            abort_all = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        for i in range(iterations):
            if abort_all:
                break
                
            stop_recording = False
            logger.info(f"--- Iteration {i+1}/{iterations} ---")
                
            try:
                # Delegate the safe subprocess execution to the existing utility
                target_wad_path = generator.build_map("temp_random.wad")
            except Exception as e:
                logger.error(f"ObligeGenerator failed: {e}")
                continue
            
            # 2. Init ViZDoom with the newly generated WAD
            game = get_game(cfg_path, target_wad_path, all_sensors, mode=vizdoom.Mode.SPECTATOR, map_name="map01")
            game.init()
            
            for key, command in active_bindings.items():
                game.send_game_command(f"bind {key} {command}")
                
            frames, depths, audios, thermals, actions = [], [], [], [], []
            
            game.new_episode()
            last_known_buffers = {}
            
            logger.info(f"Recording for {duration_secs} seconds...")
            
            while not game.is_episode_finished() and not stop_recording:
                if game.get_episode_time() >= max_tics:
                    break
                    
                state = game.get_state()
                if state is None:
                    game.advance_action()
                    continue

                action = game.get_last_action()
                extracted = SensoryExtractor.get_numpy_state(state, all_sensors)
                
                for mod in ['visual', 'depth', 'audio', 'thermal']:
                    if mod in extracted:
                        last_known_buffers[mod] = extracted[mod]
                    elif mod in last_known_buffers:
                        extracted[mod] = last_known_buffers[mod]
                        
                if len(last_known_buffers) < 4:
                    game.advance_action()
                    continue
                    
                frames.append(extracted['visual'])
                depths.append(extracted['depth'])
                audios.append(extracted['audio'])
                thermals.append(extracted['thermal'])
                actions.append(action)
                
                game.advance_action()

            game.close()

            # 3. Save Data
            if len(frames) > 0:
                # Drop idle frames if stopped by tab
                if stop_recording and not abort_all:
                    drop_frames = 70
                    if len(frames) > drop_frames:
                        logger.info(f"Map skipped. Truncating the last {drop_frames} frames to remove idle time...")
                        frames = frames[:-drop_frames]
                        actions = actions[:-drop_frames]
                        depths = depths[:-drop_frames]
                        audios = audios[:-drop_frames]
                        thermals = thermals[:-drop_frames]
                    else:
                        logger.warning("Map killed too early. Discarding frames.")
                        frames, actions, depths, audios, thermals = [], [], [], [], []
                
                if frames:
                    output_path = get_unique_filename(output_dir, file_prefix, "npz")
                    logger.info(f"Saving {len(frames)} frames to {output_path}...")
                    
                    save_dict = {
                        'frames': np.array(frames), 
                        'actions': np.array(actions),
                        'depths': np.array(depths),
                        'audios': np.array(audios),
                        'thermals': np.array(thermals)
                    }
                    np.savez_compressed(output_path, **save_dict)
            else:
                logger.warning("No frames collected in this iteration.")

    except Exception as e:
        logger.error(f"Randomize pipeline interrupted: {e}", exc_info=True)
    finally:
        listener.stop()