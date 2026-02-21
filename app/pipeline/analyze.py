"""
Analysis Module: Diagnostics and Validation.

This module provides tools for inspecting the integrity of the ETL pipeline's 
output data and auditing the performance of trained models. It ensures datasets 
are balanced and normal, and generates precision/recall matrices to evaluate 
model convergence.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from jinja2 import Environment, FileSystemLoader

from app.models.config import GolemConfig
from app.models.dataset import DoomStreamingDataset
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, register_command, get_latest_parameters

logger = logging.getLogger(__name__)

@register_command("inspect")
def inspect(cfg: GolemConfig, target_file: str = None):
    r"""
    Analyzes a training dataset file for shape integrity and class balance.

    This function loads a specific ``.npz`` recording and validates that the 
    visual frames are properly normalized. It also aggregates the action vectors 
    to report the distribution of actions taken, specifically flagging high 
    "idle time" which can cause the network to converge to inaction due to 
    class imbalance.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        target_file (str, optional): The specific filename to inspect. If ``None``, 
            it automatically loads the most recently generated data file for the 
            currently active profile. Default: ``None``.
    """
    active_profile = cfg.brain.mode
    prefix_clean = cfg.data.prefix.rstrip('_')
    
    if target_file:
        file_path = Path(target_file)
        if not file_path.is_absolute():
             file_path = Path(resolve_path(target_file))
    else:
        data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
        
        files = list(data_dir.glob(f"{prefix_clean}_*.npz"))
        
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

@register_command("audit")
def audit(cfg: GolemConfig, module_name: str = "all"):
    r"""
    Runs a diagnostic brain scan to evaluate the active model's predictive accuracy.

    This function performs a forward pass on a subset of the dataset (up to 50 batches) 
    without updating the model weights. It compares the model's action probabilities 
    against the ground-truth human actions and calculates the Precision, Recall, 
    and Support for each action class. This is critical for identifying whether the 
    agent is successfully learning rare actions (like shooting) or if it has fallen 
    into a convergence trap.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): The specific module to audit against 
            (e.g., "combat", "navigation"). If "all", it evaluates against all 
            available data for the active profile. Default: ``"all"``.
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")

    # 1. Load Data
    active_profile = cfg.brain.mode
    data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    prefix_clean = cfg.data.prefix.rstrip('_')
    
    if module_name and module_name.lower() == "all":
        file_pattern = f"{prefix_clean}_*.npz"
    else:
        file_pattern = f"{prefix_clean}_{module_name}*.npz"

    dataset = DoomStreamingDataset(str(data_dir), seq_len=32, file_pattern=file_pattern)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Discover Brain Architecture & Load Model
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    
    # Base defaults
    cortical_depth = cfg.brain.cortical_depth
    working_memory = cfg.brain.working_memory
    
    # Intelligently discover actual parameters from the latest archive
    archives = list(model_dir.glob("*.pth"))
    cortical_depth, working_memory = get_latest_parameters(archives)
    
    try:
        # Load state dict first to intelligently resolve n_actions and avoid tensor mismatches 
        # caused by runtime config overrides not updating the action space size.
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        
        n_actions = cfg.training.action_space_size
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]
            
        model = DoomLiquidNet(
            n_actions=n_actions,
            cortical_depth=cortical_depth,
            working_memory=working_memory,
            sensors=cfg.brain.sensors
        ).to(device)
        
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        logger.error(f"No brain found at {active_model_path}. Train first!")
        return

    # 3. Scan
    logger.info(f"Scanning neural pathways (Module: {module_name})...")
    
    all_preds = []
    all_targets = []
    max_batches = 50 
    
    with torch.no_grad():
        for i, (inputs, actions) in enumerate(dataloader):
            if i >= max_batches: 
                break
            
            x_vis = inputs['visual'].to(device)
            x_aud = inputs['audio'].to(device) if 'audio' in inputs else None
            
            logits, _ = model(x_vis, x_aud=x_aud)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            targets = actions.cpu().numpy()
            
            # Use dynamic n_actions rather than potentially stale cfg.training.action_space_size
            all_preds.append(preds.reshape(-1, n_actions))
            all_targets.append(targets.reshape(-1, n_actions))

    if not all_preds:
        logger.error("No data found to audit!")
        return

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # 4. Report via Jinja2
    exact_acc = accuracy_score(y_true, y_pred)
    action_names = list(cfg.training.action_names)
    
    # Pad action names if the runtime config didn't dynamically expand
    if len(action_names) < n_actions:
        action_names += [f"ACTION_{i}" for i in range(len(action_names), n_actions)]
        
    metrics = []

    for i, name in enumerate(action_names):
        if i >= y_true.shape[1]:
            break
            
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        support = int(true_col.sum())
        
        tp = ((true_col == 1) & (pred_col == 1)).sum()
        fp = ((true_col == 0) & (pred_col == 1)).sum()
        fn = ((true_col == 1) & (pred_col == 0)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        
        metrics.append({
            "name": name,
            "precision": precision,
            "recall": recall,
            "support": support
        })

    env = Environment(loader=FileSystemLoader(resolve_path("app/templates")))
    template = env.get_template("audit.j2")
    
    print(template.render(
        module_name=module_name,
        sample_count=len(y_true),
        exact_acc=exact_acc,
        metrics=metrics
    ))