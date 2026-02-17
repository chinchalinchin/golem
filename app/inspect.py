import numpy as np
import logging
from pathlib import Path
from app.config import GolemConfig
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def inspect_data(cfg: GolemConfig, target_file: str = None):
    """Analyzes the shape and class balance of a training file."""
    
    # Determine target file
    if target_file:
        file_path = Path(target_file)
        if not file_path.is_absolute():
             file_path = Path(resolve_path(target_file))
    else:
        # Default to finding the latest file in data dir
        data_dir = Path(resolve_path(cfg.data.output_dir))
        files = list(data_dir.glob(f"{cfg.data.filename_prefix}*.npz"))
        if not files:
            logger.error(f"No data files found in {data_dir}")
            return
        # Sort by modification time, newest first
        file_path = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Analyzing: {file_path.name}")
    
    try:
        data = np.load(str(file_path))
        frames = data['frames']
        actions = data['actions']
        
        # Frames Analysis
        logger.info(f"Frames Shape: {frames.shape}")
        logger.info(f"Frames Range: {frames.min():.2f} - {frames.max():.2f}")
        
        if frames.max() > 1.0:
            logger.warning("Frames are NOT normalized (0-255). Expecting 0-1.")
        else:
            logger.info("Frames are normalized.")

        # Actions Analysis
        logger.info(f"Actions Shape: {actions.shape}")
        
        total_frames = len(actions)
        if total_frames == 0:
            logger.warning("Dataset is empty.")
            return

        total_presses = np.sum(actions, axis=0)
        labels = ["Left", "Right", "Attack"]
        
        for i, label in enumerate(labels):
            if i < len(total_presses):
                count = int(total_presses[i])
                pct = count / total_frames
                logger.info(f"Action '{label}': {count} ({pct:.1%})")
        
        non_action_frames = np.sum(~actions.any(axis=1))
        idle_pct = non_action_frames / total_frames
        logger.info(f"Idling: {non_action_frames} ({idle_pct:.1%})")

        if idle_pct > 0.5:
            logger.warning("High idle time detected. Model may converge to inaction.")
            
    except Exception as e:
        logger.error(f"Failed to inspect data: {e}", exc_info=True)