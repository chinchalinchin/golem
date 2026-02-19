"""
Audit Module: Brain Scan.
"""
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from jinja2 import Environment, FileSystemLoader

from app.config import GolemConfig
from app.dataset import DoomStreamingDataset
from app.brain import DoomLiquidNet
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def audit_agent(cfg: GolemConfig, module_name: str = "all"):
    """
    Runs inference on a subset of data and reports metrics.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. Load Data
    data_dir = resolve_path(cfg.data.output_dir)
    if module_name and module_name.lower() == "all":
        file_pattern = f"{cfg.data.filename_prefix}*.npz"
    else:
        file_pattern = f"{cfg.data.filename_prefix}_{module_name}*.npz"

    dataset = DoomStreamingDataset(data_dir, seq_len=32, file_pattern=file_pattern)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Load Brain
    model_path = resolve_path(cfg.training.model_save_path)
    n_actions = cfg.training.action_space_size
    
    try:
        model = DoomLiquidNet(n_actions=n_actions).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        logger.error("No brain found. Train first!")
        return

    # 3. Scan
    logger.info(f"Scanning neural pathways (Module: {module_name})...")
    
    all_preds = []
    all_targets = []
    max_batches = 50 
    
    with torch.no_grad():
        for i, (frames, actions) in enumerate(dataloader):
            if i >= max_batches: break
            frames = frames.to(device)
            logits, _ = model(frames)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            targets = actions.cpu().numpy()
            
            all_preds.append(preds.reshape(-1, n_actions))
            all_targets.append(targets.reshape(-1, n_actions))

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