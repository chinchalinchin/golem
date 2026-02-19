"""
Audit Module: Brain Scan.
Performs static analysis on the trained model against the dataset.
"""
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

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
    if module_name == "all":
        file_pattern = f"{cfg.data.filename_prefix}*.npz"
    else:
        file_pattern = f"{cfg.data.filename_prefix}_{module_name}*.npz"

    dataset = DoomStreamingDataset(data_dir, seq_len=32, file_pattern=file_pattern)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Load Brain
    model_path = resolve_path(cfg.training.model_save_path)
    n_actions = cfg.training.action_space_size
    
    try:
        model = DoomLiquidNet(n_actions=n_actions).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Loaded Brain from {model_path}")
    except FileNotFoundError:
        logger.error("No brain found. Train first!")
        return

    # 3. Scan (Run one large batch)
    logger.info("Scanning neural pathways...")
    
    all_preds = []
    all_targets = []
    
    max_batches = 10 # Don't scan everything, just a sample
    with torch.no_grad():
        for i, (frames, actions) in enumerate(dataloader):
            if i >= max_batches: break
            
            frames = frames.to(device)
            logits = model(frames)
            probs = torch.sigmoid(logits)
            
            # Threshold at 0.5
            preds = (probs > 0.5).float().cpu().numpy()
            targets = actions.cpu().numpy()
            
            # Flatten: (Batch, Time, Actions) -> (Samples, Actions)
            all_preds.append(preds.reshape(-1, n_actions))
            all_targets.append(targets.reshape(-1, n_actions))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # 4. Report
    action_names = [
        "Move Fwd", "Move Back", "Move Left", "Move Right", 
        "Turn Left", "Turn Right", "Attack", "Use"
    ]
    
    print("\n" + "="*60)
    print(f"GOLEM BRAIN AUDIT | Data: {module_name} | Samples: {len(y_true)}")
    print("="*60)
    
    # Calculate simple accuracy (Exact Match)
    # This is harsh: if you miss ONE button in the chord, it's wrong.
    exact_acc = accuracy_score(y_true, y_pred)
    print(f"Exact Sequence Match: {exact_acc:.1%}")
    print("-" * 60)
    
    # Per-Channel Analysis
    print(f"{'ACTION':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'SUPPORT':<10}")
    print("-" * 60)
    
    for i, name in enumerate(action_names):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        
        # Support: How many times did the Human press this?
        support = int(true_col.sum())
        
        # True Positives
        tp = ((true_col == 1) & (pred_col == 1)).sum()
        # False Positives (Hallucinations)
        fp = ((true_col == 0) & (pred_col == 1)).sum()
        # False Negatives (Missed Reactions)
        fn = ((true_col == 1) & (pred_col == 0)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        
        print(f"{name:<15} | {precision:.1%}      | {recall:.1%}      | {support:<10}")

    print("="*60 + "\n")