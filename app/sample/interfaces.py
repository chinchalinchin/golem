# Standard Libraries
import os
import random
import logging
import subprocess
import shutil

# Application Libraries
from app.utils.conf import resolve_path

from pathlib import Path

from app.models.config import RandomizerConfig
from app.utils.conf import resolve_path


logger = logging.getLogger(__name__)


class ObligeGenerator:
    def __init__(self, cfg: RandomizerConfig):
        # We ensure the output directory is an absolute path for Docker volume mounting
        self.output_dir = Path(resolve_path(cfg.output)).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dump the Pydantic model into a dict so we can sample from it
        self.base_oblige_config = cfg.oblige.model_dump()

    def build_map(self, filename: str = "golem_procgen.wad", overrides: dict = None) -> str:
        """Compiles the map using the containerized Oblige engine with randomized parameters."""
        target_wad_absolute = str(self.output_dir / filename)
        temp_wad_name = "temp_batch.wad" 
        container_output_dir = "/output"

        args = [
            "docker", "run", "--rm",
            "-v", f"{self.output_dir}:{container_output_dir}",
            "golem-oblige:latest",
            "--batch", f"{container_output_dir}/{temp_wad_name}"
        ]
        active_params = {}
        for key, value in self.base_oblige_config.items():
            # If an override is provided for this key, force it. Otherwise, sample randomly.
            if overrides and key in overrides:
                chosen_val = overrides[key]
            else:
                chosen_val = random.choice(value) if isinstance(value, list) else value
                
            active_params[key] = chosen_val
            args.append(f"{key}={chosen_val}")
            
        logger.info(f"Compiling procedural map via Docker with parameters: {active_params}")
        
        try:
            subprocess.run(args, check=True, capture_output=True, text=True)
            
            compiled_temp_path = self.output_dir / temp_wad_name
            if compiled_temp_path.exists():
                shutil.move(str(compiled_temp_path), target_wad_absolute)
                return target_wad_absolute
            else:
                raise FileNotFoundError("Docker returned success, but the temporary WAD file is missing.")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Oblige container failed with exit code {e.returncode}.")
            logger.error(f"Docker STDOUT:\n{e.stdout}") 
            logger.error(f"Docker STDERR:\n{e.stderr}")
            raise