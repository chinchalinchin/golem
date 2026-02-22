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
from app.utils import resolve_path, register_command, apply_latest_parameters

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
def audit(cfg: GolemConfig, module_name: str = "all", full: bool = False):
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
        full (bool, optional): If ``True``, evaluates the entire dataset instead 
            of capping at 50 sequence batches. Default: ``False``
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
        dsp_config=cfg.brain.dsp
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Discover Brain Architecture & Load Model
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
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
            dsp_config=cfg.brain.dsp  # <--- New
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

@register_command("summary")
def summary(cfg: GolemConfig, module_name: str = None):
    r"""
    Prints a detailed architectural summary of the active brain configuration.

    This function instantiates the LNN based on the current configuration and uses 
    the `torchinfo` package to perform a dummy forward pass. It displays the exact 
    tensor dimensions at each layer, the parameter counts, and validates that 
    the multi-modal sensor fusion layers are properly scaling and concatenating 
    into the Liquid Core.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): Ignored. Included for CLI compatibility.
    """
    try:
        import torchinfo
    except ImportError:
        logger.error("torchinfo is required for the summary command. Run: pip install torchinfo")
        return

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 1. Base defaults
    active_profile = cfg.brain.mode
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    n_actions = cfg.training.action_space_size 

    # 2. Discover architecture from archives
    archives = list(model_dir.glob("*.pth"))
    apply_latest_parameters(cfg, archives)
        
    # 3. Discover action space and load state dict (if it exists)
    if active_model_path.exists():
        try:
            state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
            if 'output.weight' in state_dict:
                n_actions = state_dict['output.weight'].shape[0]
        except Exception as e:
            logger.warning(f"Could not load state dict from {active_model_path}: {e}")

    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory,
        sensors=cfg.brain.sensors,
        dsp_config=cfg.brain.dsp
    ).to(device)

    # 4. Construct Multi-Modal Dummy Tensors
    seq_len = cfg.training.sequence_length
    batch_size = 1 
    
    c_vis = 4 if cfg.brain.sensors.depth else 3
    x_vis = torch.randn(batch_size, seq_len, c_vis, 64, 64).to(device)
    
    x_aud = None
    if cfg.brain.sensors.audio:
        # Calculate raw audio samples per frame (44100 Hz / 35 FPS = 1260)
        audio_samples_per_frame = int(cfg.brain.dsp.sample_rate / 35)
        # Dummy tensor now represents raw waveforms, not spectrograms
        x_aud = torch.randn(batch_size, seq_len, 2, audio_samples_per_frame).to(device)
        
    x_thm = None
    if cfg.brain.sensors.thermal:
        x_thm = torch.randn(batch_size, seq_len, 1, 64, 64).to(device)
        
    # Strip None values to prevent torchinfo memory calculation crashes
    input_dict = {"x_vis": x_vis}
    if x_aud is not None:
        input_dict["x_aud"] = x_aud
    if x_thm is not None:
        input_dict["x_thm"] = x_thm

    logger.info("======================================================")
    logger.info(f"Generating Architectural Summary for Profile: {active_profile.upper()}")
    logger.info(f"Sensors Enabled -> Visual: True | Depth: {cfg.brain.sensors.depth} | Audio: {cfg.brain.sensors.audio} | Thermal: {cfg.brain.sensors.thermal}")
    logger.info("======================================================")
    
    torchinfo.summary(
        model, 
        input_data=input_dict, 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3
    )