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
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def inspect_data(cfg: GolemConfig, target_file: str = None):
    """Analyzes the shape and class balance of a training file."""
    
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


def audit_agent(cfg: GolemConfig, module_name: str = "all"):
    """
    Runs inference on a subset of data and reports metrics.
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

    # 2. Load Brain
    model_path = Path(resolve_path(cfg.data.dirs["training"])) / "golem.pth"
    
    try:
        model = DoomLiquidNet(
            n_actions=cfg.training.action_space_size,
            cortical_depth=cfg.brain.cortical_depth,
            working_memory=cfg.brain.working_memory
        ).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error(f"No brain found at {model_path}. Train first!")
        return

    # 3. Scan
    logger.info(f"Scanning neural pathways (Module: {module_name})...")
    
    all_preds = []
    all_targets = []
    max_batches = 50 
    
    with torch.no_grad():
        for i, (frames, actions) in enumerate(dataloader):
            if i >= max_batches: 
                break
            
            frames = frames.to(device)
            logits, _ = model(frames)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            targets = actions.cpu().numpy()
            
            all_preds.append(preds.reshape(-1, cfg.training.action_space_size))
            all_targets.append(targets.reshape(-1, cfg.training.action_space_size))

    if not all_preds:
        logger.error("No data found to audit!")
        return

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # 4. Report via Jinja2
    exact_acc = accuracy_score(y_true, y_pred)
    action_names = cfg.training.action_names
    metrics = []

    for i, name in enumerate(action_names):
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