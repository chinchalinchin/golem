"""
Procedural Generation Module: Oblige Wrapper.
Generates randomized DOOM maps to cure spatial overfitting and covariate shift.
"""
import os
import random
import logging
import subprocess
from pathlib import Path

from app.models.config import GolemConfig
from app.utils.conf import resolve_path, register_command
from app.pipeline.record import record  # We will hand off to the recorder

logger = logging.getLogger(__name__)

class ObligeGenerator:
    def __init__(self, executable_path: str, output_dir: str):
        self.executable = resolve_path(executable_path)
        self.output_dir = Path(resolve_path(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Oblige executable not found at: {self.executable}")

    def randomize_config(self) -> dict:
        """Returns a randomized dictionary of Oblige Lua parameters."""
        return {
            "game": "doom2",
            "length": "single",
            "theme": random.choice(["tech", "urban", "hell", "jumbled", "original"]),
            "size": random.choice(["micro", "tiny", "small"]), # Keep maps small to prevent dead states
            "outdoors": random.choice(["none", "some", "plenty"]),
            "caves": random.choice(["none", "some", "plenty"]),
            "mons": random.choice(["normal", "lots", "nuts"]), # High threat density
            "health": random.choice(["less", "normal", "more"]),
            "weapons": random.choice(["soon", "normal"]),
            "secret_rooms": "none" # Disable secrets so the agent doesn't get stuck searching
        }

    def build_map(self, filename: str = "golem_procgen.wad") -> str:
        """Compiles the map using the headless CLI."""
        target_wad = str(self.output_dir / filename)
        config = self.randomize_config()
        
        # Construct the CLI arguments
        args = [self.executable, "--batch", target_wad]
        for key, value in config.items():
            args.append(f"{key}={value}")
            
        logger.info(f"Compiling procedural map with parameters: {config}")
        
        try:
            # Run headless compilation
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Map compiled successfully: {target_wad}")
            return target_wad
        except subprocess.CalledProcessError as e:
            logger.error(f"Oblige compilation failed: {e}")
            raise

@register_command("generate")
def generate(cfg: GolemConfig, episodes: int = 5):
    """
    Generates a random procedural map and immediately launches a recording session.
    """
    try:
        oblige_exe = cfg.randomizer.executable  # Assuming you added this to app.yaml
        wad_dir = cfg.randomizer.output
    except AttributeError:
        logger.error("Oblige configuration missing from app.yaml.")
        return

    generator = ObligeGenerator(oblige_exe, wad_dir)
    generated_wad = generator.build_map("golem_temp.wad")

    # --- Dynamic Handoff to Recording ---
    # To properly hook this up, we need to inject the generated_wad path into
    # your ViZDoom initialization logic before calling the recorder.
    
    logger.info("Injecting procedural WAD into recording pipeline...")
    
    # We temporarily override the 'basic' module config to point to our new procedural map
    cfg.modules["basic"].scenario = generated_wad 
    cfg.modules["basic"].map = "map01"
    cfg.modules["basic"].episodes = int(episodes)
    
    # Call your existing record function (which now supports truncation via TAB)
    record(cfg, module_name="basic")