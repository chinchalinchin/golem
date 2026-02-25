# Standard Libraries
import os
import logging
import subprocess
import shutil

# External Libraries
import vizdoom

# Application Libraries
from app.utils.conf import resolve_path

from pathlib import Path

from app.models.config import RandomizerConfig
from app.utils.conf import resolve_path


logger = logging.getLogger(__name__)


class ObligeGenerator:
    def __init__(self, cfg: RandomizerConfig):
        self.executable = resolve_path(cfg.executable)
        self.output_dir = Path(resolve_path(cfg.output))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.oblige = cfg.oblige

        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Oblige executable not found at: {self.executable}")

    def build_map(self, filename: str = "golem_procgen.wad") -> str:
        """Compiles the map using the headless CLI."""
        target_wad_absolute = str(self.output_dir / filename)
        temp_wad_name = "temp_batch.wad" 

        # HACK: There is an issue with how Oblige discovered the working directory of the executable
        #       when it is called with an absolute path. This hack emulates the terminal exactly by 
        #       forcing argv[0] to be "./Oblige" instead of the absolute path.
        args = ["./Oblige", "--batch", temp_wad_name]
        for key, value in self.oblige.items():
            args.append(f"{key}={value}")
            
        logger.info(f"Compiling procedural map with parameters: {self.oblige}")
        
        try:
            oblige_dir = os.path.dirname(self.executable)
            subprocess.run(args, check=True, capture_output=True, text=True, cwd=oblige_dir)
            
            compiled_temp_path = os.path.join(oblige_dir, temp_wad_name)
            if os.path.exists(compiled_temp_path):
                shutil.move(compiled_temp_path, target_wad_absolute)
                logger.info(f"Map compiled and moved successfully: {target_wad_absolute}")
                return target_wad_absolute
            else:
                raise FileNotFoundError("Oblige returned success, but the temporary WAD file is missing.")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Oblige compilation failed with exit code {e.returncode}.")
            logger.error(f"Oblige STDOUT:\n{e.stdout}") 
            logger.error(f"Oblige STDERR:\n{e.stderr}")
            raise

def get_scenario(scenario_name: str) -> str:
    """
    Locates a ViZDoom scenario WAD file.
    """
    if os.path.basename(scenario_name) == scenario_name:
        package_path = os.path.dirname(vizdoom.__file__)
        scenario_path = os.path.join(package_path, "scenarios", scenario_name)
        if os.path.exists(scenario_path):
            return scenario_path

    local_path = resolve_path(scenario_name)
    if os.path.exists(local_path):
        return local_path
        
    package_path = os.path.dirname(vizdoom.__file__)
    scenario_path = os.path.join(package_path, "scenarios", scenario_name)
    logger.warning(f"Could not find scenario. Checked:\n  {scenario_path}\n  {local_path}")
        
    return scenario_path


def get_game(pth: str, scenario: str, sensors=None, mode=vizdoom.Mode.PLAYER, map_name=None) -> vizdoom.DoomGame:
    """
    Retrieves a ViZDoom DoomGame instances configured for Golem training.
    """
    cfg_path = resolve_path(pth)
    scenario_path = get_scenario(scenario)
    
    logger.info(f"Loading ViZDoom Config: {cfg_path}")
    logger.info(f"Loading ViZDoom Scenario: {scenario_path}")
    logger.info(f"Initializing Engine in Mode: {mode.name}")
    
    game = vizdoom.DoomGame()
    game.load_config(cfg_path)    
    game.set_doom_scenario_path(scenario_path)
    
    # Inject the runtime map override if provided
    if map_name:
        game.set_doom_map(map_name)
        
    game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    
    # Dynamically set the mode (Defaults to PLAYER for inference, SPECTATOR for recording)
    game.set_mode(mode) 
    
    # Enforce HUD rendering to satisfy the 'Doomguy' heuristic requirement
    game.set_render_hud(True)
    
    if sensors:
        logger.info(f"Configuring Sensors - Depth: {getattr(sensors, 'depth', False)}, Audio: {getattr(sensors, 'audio', False)}, Thermal: {getattr(sensors, 'thermal', False)}")
        
        if getattr(sensors, 'depth', False):
            game.set_depth_buffer_enabled(True)
        if getattr(sensors, 'audio', False):
            game.set_audio_buffer_enabled(True)
        if getattr(sensors, 'thermal', False):
            game.set_labels_buffer_enabled(True)  
    return game

