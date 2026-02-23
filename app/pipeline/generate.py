"""
Procedural Generation Module: Oblige Wrapper.
Generates randomized DOOM maps to cure spatial overfitting and covariate shift.
"""
import os
import random
import logging
import subprocess
import shutil
from pathlib import Path

from app.models.config import GolemConfig
from app.utils.conf import resolve_path, register_command
from app.pipeline.record import record

logger = logging.getLogger(__name__)

class ObligeGenerator:
    def __init__(self, executable_path: str, output_dir: str):
        self.executable = resolve_path(executable_path)
        self.output_dir = Path(resolve_path(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Oblige executable not found at: {self.executable}")

    def randomize_config(self) -> dict:
        """Returns a randomized dictionary of strictly valid Oblige 7.70 Lua parameters."""
        return {
            "game": "doom2",
            "engine": "zdoom",  # FIX: ViZDoom is based on ZDoom
            "length": "single", # 'single', 'episode', or 'game'
            "theme": random.choice(["original", "tech", "urban", "hell", "jumbled", "mixed"]),
            "size": "small",    # FIX: 'micro' and 'tiny' are invalid in 7.70. Use 'small' or 'mixed'
            "outdoors": random.choice(["none", "mixed", "plenty"]),
            "caves": random.choice(["none", "mixed", "plenty"]),
            "liquids": "none",  # Prevent agent from getting stuck in damaging acid pits early on
            "hallways": "mixed",
            "teleporters": "none", # Teleporters cause severe spatial discontinuity for LNNs, disable them
            "steepness": "mixed",
            "mons": "normal", # "random.choice(["normal", "lots", "nuts"])",
            "strength": "medium",
            "health": "normal",
            "ammo": "normal",
            "weapons": "normal"
        }

    def build_map(self, filename: str = "golem_procgen.wad") -> str:
        """Compiles the map using the headless CLI."""
        target_wad_absolute = str(self.output_dir / filename)
        temp_wad_name = "temp_batch.wad" 
        
        config = self.randomize_config()
        
        # FIX: Emulate the terminal exactly. Force argv[0] to be "./Oblige" instead of the absolute path.
        args = ["./Oblige", "--batch", temp_wad_name]
        for key, value in config.items():
            args.append(f"{key}={value}")
            
        logger.info(f"Compiling procedural map with parameters: {config}")
        
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

@register_command("generate")
def generate(cfg: GolemConfig, episodes: int = 1):
    """
    Generates a random procedural map and immediately launches a recording session.
    """

    generator = ObligeGenerator(cfg.randomizer.executable, cfg.randomizer.output)
    generated_wad = generator.build_map("temp.wad")
    
    logger.info("Injecting procedural WAD into recording pipeline...")
    
    # We dynamically inject the newly compiled WAD into the "basic" module slot
    cfg.modules["basic"].scenario = generated_wad 
    cfg.modules["basic"].map = "map01"
    cfg.modules["basic"].episodes = episodes
    
    # Boot the standard recording loop (which includes your new TAB-to-truncate feature)
    record(cfg, module_name="basic")