# Standard Libraries
import logging
from pathlib import Path

# External Libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from jinja2 import Environment, FileSystemLoader

# Application Libraries
from app.models.config import GolemConfig
from app.models.dataset import DoomStreamingDataset
from app.models.brain import DoomLiquidNet
from app.utils.conf import resolve_path, register_command
from app.utils.model import apply_latest_parameters


logger = logging.getLogger(__name__)


@register_command("audit")
def audit(cfg: GolemConfig, module_name: str = "all", full: bool = False, target_file: str = None):
    r"""
    Runs a diagnostic brain scan to evaluate the active model's predictive accuracy.

    This function performs a forward pass on a subset of the dataset (up to 50 batches) without updating the model weights. It compares the model's action probabilities against the ground-truth human actions and calculates the Precision, Recall, and Support for each action class. This is critical for identifying whether the agent is successfully learning rare actions (like shooting) or if it has fallen into a convergence trap.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): The specific module to audit against 
            (e.g., "combat", "navigation"). If "all", it evaluates against all 
            available data for the active profile. Default: ``"all"``.
        full (bool, optional): If ``True``, evaluates the entire dataset instead 
            of capping at 50 sequence batches. Default: ``False``
        target_file (str, optional): The specific model file to load.
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

    dataset = DoomStreamingDataset(
        str(data_dir), 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern,
        augment=cfg.training.augmentation.mirror,
        action_names=cfg.training.action_names,
        dsp_config=cfg.brain.dsp,
        sensors=cfg.brain.sensors
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Discover Brain Architecture & Load Model
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    
    if target_file:
        active_model_path = model_dir / target_file
        if not active_model_path.exists():
            logger.error(f"Target model file not found: {active_model_path}")
            return
        archives = [active_model_path]
    else:
        active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
        # Intelligently discover actual parameters from the latest archive
        archives = list(model_dir.glob("*.pth"))
    
    apply_latest_parameters(cfg, archives)
    
    try:
        # Load state dict first to intelligently resolve n_actions and avoid tensor mismatches 
        # caused by runtime config overrides not updating the action space size.
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        
        n_actions = cfg.training.action_space_size
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]
            
        model = DoomLiquidNet(
            n_actions=n_actions,
            cortical_depth=cfg.brain.cortical_depth,
            working_memory=cfg.brain.working_memory,
            sensors=cfg.brain.sensors,
            dsp_config=cfg.brain.dsp
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
    max_batches = float('inf') if full else 50
    
    with torch.no_grad():
        for i, (inputs, actions) in enumerate(dataloader):
            if i >= max_batches: 
                break
            
            x_vis = inputs['visual'].to(device)
            x_aud = inputs['audio'].to(device) if 'audio' in inputs else None
            x_thm = inputs['thermal'].to(device) if 'thermal' in inputs else None
            
            logits, _ = model(x_vis, x_aud=x_aud, x_thm=x_thm)
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

