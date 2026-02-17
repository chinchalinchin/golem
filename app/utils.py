import os
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Returns the absolute path to the project root (one level up from app)."""
    # Assumes utils.py is in /lib/golem/app/
    return Path(__file__).parent.parent.resolve()

def resolve_path(path_str: str) -> str:
    """Resolves a path relative to the project root."""
    root = get_project_root()
    return str(root / path_str)

def get_unique_filename(directory: Union[str, Path], prefix: str, extension: str = "npz") -> str:
    """Generates a unique filename (e.g., data_1.npz) to prevent overwrites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        filename = f"{prefix}_{counter}.{extension}"
        full_path = directory / filename
        if not full_path.exists():
            return str(full_path)
        counter += 1

def setup_logging(level_str: str = "INFO"):
    """Configures the root logger."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )