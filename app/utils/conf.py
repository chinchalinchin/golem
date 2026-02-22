from pathlib import Path
import logging
from typing import Callable, Union

COMMAND_REGISTRY = {}

def setup_logging(level_str: str = "INFO"):
    """Configures the root logger with a standard format."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def register_command(name: str = None) -> Callable:
    """
    Decorator to register a CLI command into the global registry.
    """
    def decorator(func: Callable) -> Callable:
        cmd_name = name if name else func.__name__
        COMMAND_REGISTRY[cmd_name] = func
        return func
    return decorator


def get_project_root() -> Path:
    """Returns the absolute path to the project root (one level up from app)."""
    return Path(__file__).parent.parent.parent.resolve()

def resolve_path(path_str: str) -> str:
    """
    Resolves a path relative to the project root.
    If the path is already absolute, it is returned as-is.
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
        
    root = get_project_root()
    return str(root / path)

def get_unique_filename(directory: Union[str, Path], prefix: str, extension: str = "npz") -> str:
    """Generates a unique filename (e.g., data_1.npz) to prevent overwrites."""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = Path(resolve_path(str(directory)))
        
    directory.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        filename = f"{prefix}.{counter}.{extension}"
        full_path = directory / filename
        if not full_path.exists():
            return str(full_path)
        counter += 1
