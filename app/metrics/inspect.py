"""
Analysis Module: Diagnostics and Validation.

This module provides tools for inspecting the integrity of the ETL pipeline's output data and auditing the performance of trained models. It ensures datasets are balanced and normal, and generates precision/recall matrices to evaluate model convergence.
"""
# Standard Libraries
import logging
from pathlib import Path

# External Libraries
import numpy as np
from jinja2 import Environment, FileSystemLoader

# Application Libraries
from app.models.config import GolemConfig
from app.utils.conf import resolve_path, register_command


logger = logging.getLogger(__name__)


@register_command("inspect")
def inspect(cfg: GolemConfig, target_file: str = None):
    r"""
    Analyzes a training dataset file for shape integrity and class balance.

    This function loads a specific ``.npz`` recording and validates that the visual frames are properly normalized. It also aggregates the action vectors to report the distribution of actions taken, specifically flagging high "idle time" which can cause the network to converge to inaction due to class imbalance.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        target_file (str, optional): The specific filename to inspect. If ``None``, 
            it automatically loads the most recently generated data file for the 
            currently active profile. Default: ``None``.
    """
    active_profile = cfg.brain.mode
    
    if target_file:
        file_path = Path(target_file)
        if not file_path.is_absolute():
             file_path = Path(resolve_path(target_file))
    else:
        data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
        
        files = list(data_dir.glob(f"{cfg.data.prefix}_*.npz"))
        
        if not files:
            logger.error(f"No data files found in {data_dir}")
            return
        file_path = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Analyzing: {file_path.name}")
    
    try:
        data = np.load(str(file_path))
        frames = data['frames']
        actions = data['actions']
        
        total_frames = len(actions)
        if total_frames == 0:
            logger.warning("Dataset is empty.")
            return

        total_presses = np.sum(actions, axis=0)
        labels = cfg.training.action_names
        action_counts = []
        
        for i, label in enumerate(labels):
            if i < len(total_presses):
                count = int(total_presses[i])
                pct = count / total_frames
                action_counts.append({"label": label, "count": count, "pct": pct})
        
        non_action_frames = np.sum(~actions.any(axis=1))
        idle_pct = non_action_frames / total_frames
        
        # Render Report
        env = Environment(loader=FileSystemLoader(resolve_path("app/templates")))
        template = env.get_template("inspect.j2")
        
        print(template.render(
            filename=file_path.name,
            frames_shape=frames.shape,
            frames_range=(frames.min(), frames.max()),
            is_normalized=(frames.max() <= 1.0),
            actions_shape=actions.shape,
            total_frames=total_frames,
            action_counts=action_counts,
            idle_count=non_action_frames,
            idle_pct=idle_pct
        ))
            
    except Exception as e:
        logger.error(f"Failed to inspect data: {e}", exc_info=True)

