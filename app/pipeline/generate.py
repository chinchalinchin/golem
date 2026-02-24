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

from app.models.config import GolemConfig, RandomizerConfig
from app.utils.conf import resolve_path, register_command

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

@register_command("generate")
def generate(cfg: GolemConfig, target_file: str = None):
    """
    Generates a random procedural map and immediately launches a recording session.
    """
    if target_file is None:
        target_file = "temp.wad"

    generator = ObligeGenerator(cfg.randomizer)
    generator.build_map(target_file)