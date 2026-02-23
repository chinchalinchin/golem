"""
Training Module: Behavioral Cloning Loop.

This module handles the supervised learning pipeline for the Golem agent. It implements a behavioral cloning loop that maps visual sequence inputs (screen buffers) to expert action vectors using Binary Cross-Entropy loss, effectively teaching the agent to mimic human gameplay demonstrations.
"""

# Standard Libraries
import logging
import time
from datetime import datetime
from pathlib import Path

# External Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Application Libraries
from app.models.config import GolemConfig, LossType
from app.models.dataset import DoomStreamingDataset
from app.models.brain import DoomLiquidNet
from app.models.loss import FocalLossWithLogits, AsymmetricLoss
from app.utils.conf import resolve_path, get_unique_filename, register_command
from app.utils.model import apply_latest_parameters, generate_model_prefix

logger = logging.getLogger(__name__)

@register_command("train")
def train(cfg: GolemConfig, module_name: str = None, include_recovery: bool = False):
    r"""
    Trains the Liquid Neural Network using captured expert demonstrations.

    This function orchestrates the dataset streaming and the model's training loop. It dynamically selects the best available hardware accelerator (CUDA, MPS, or CPU), initializes the dataset with optional mirror augmentation, and optimizes the network weights using the Adam optimizer.

    If an active model already exists for the current profile (e.g., ``fluid``), it loads the existing weights to perform continuous fine-tuning. Upon completion, it saves the updated model to both a timestamped archive and the active profile slot.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): The specific data module to train against (e.g., "combat", "navigation"). If ``"all"`` or ``None``, it trains across all available data for the active profile (Generalization Mode). Default: ``None``.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple Metal (MPS) acceleration detected and enabled.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA acceleration detected and enabled.")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected. Training will be slow on CPU.")
    
    active_profile = cfg.brain.mode
    base_data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    prefix_clean = cfg.data.prefix.rstrip('_')
    
    if module_name and module_name.lower() != "all":
        file_pattern = f"{prefix_clean}_{module_name}*.npz"
        logger.info(f"Training restricted to module: {module_name}")
    else:
        file_pattern = f"{prefix_clean}_*.npz"        
        logger.info("Training on ALL available modules (Generalization Mode)")

    # Aggregate target directories
    data_dirs = [base_data_dir]
    if include_recovery:
        recovery_dir = base_data_dir / "recovery"
        if recovery_dir.exists():
            data_dirs.append(recovery_dir)
            logger.info("Recovery (DAgger) data will be included in this training run.")
        else:
            logger.warning(f"Recovery directory {recovery_dir} not found. Proceeding without recovery data.")

    dataset = DoomStreamingDataset(
        data_dirs, 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern,
        augment=cfg.training.augmentation.mirror,
        action_names=cfg.training.action_names,
        dsp_config=cfg.brain.dsp
    )
    
    if len(dataset) == 0:
        logger.error(f"No training data found matching pattern: {file_pattern} in {data_dirs}")
        return

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False, drop_last=True)
    
    n_actions = cfg.training.action_space_size

    # 1. Discover architecture and dimensions if resuming training
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = base_data_dir / "golem.pth"  # 
    state_dict = None
    archives = []

    if active_model_path.exists():
        logger.info(f"Discovering existing brain architecture from {active_model_path} for fine-tuning...")
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]

        archives = list(model_dir.glob("*.pth"))

    apply_latest_parameters(cfg, archives)

    # 3. Initialize dynamic model
    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory,
        sensors=cfg.brain.sensors,
        dsp_config=cfg.brain.dsp
    ).to(device)
    
    if state_dict:
        model.load_state_dict(state_dict)
    
    if cfg.training.loss == LossType.FOCAL:
        logger.info("Initializing Focal Loss with static alpha vector from configuration...")
        
        # 1. Tally raw action counts across all loaded files for logging purposes
        action_counts = np.zeros(n_actions)
        total_samples = 0
        for action_array in dataset.action_arrays:
            action_counts += np.sum(action_array, axis=0)
            total_samples += action_array.shape[0]
            
        # 2. Acknowledge Augmentation Prior: Enforce left/right symmetry
        if dataset.augment:
            for left_idx, right_idx in dataset.swap_pairs:
                avg_count = (action_counts[left_idx] + action_counts[right_idx]) / 2.0
                action_counts[left_idx] = avg_count
                action_counts[right_idx] = avg_count
                
        # 3. Construct bounded alpha tensor
        # We rely on the configured static alpha (e.g., 0.25) to prevent button-mashing
        # while letting gamma exponentially scale the loss for hard examples.
        alpha_vector = np.full(n_actions, cfg.loss.focal.alpha)
        
        alpha_tensor = torch.tensor(alpha_vector, dtype=torch.float32).to(device)
        criterion = FocalLossWithLogits(alpha=alpha_tensor, gamma=cfg.loss.focal.gamma)

    elif cfg.training.loss == LossType.BCE:
        criterion = nn.BCEWithLogitsLoss()
    
    elif cfg.training.loss == LossType.ASL:
        criterion = AsymmetricLoss(
            gamma_neg=cfg.loss.asymmetric.gamma_neg,
            gamma_pos=cfg.loss.asymmetric.gamma_pos,
            clip=cfg.loss.asymmetric.clip
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    model.train()
    hx = None # Initialize global hidden state

    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        total_loss = 0
        batches = 0
        
        for batch_idx, (inputs, actions) in enumerate(dataloader):
            x_vis = inputs['visual'].to(device)
            x_aud = inputs['audio'].to(device) if 'audio' in inputs else None
            x_thm = inputs['thermal'].to(device) if 'thermal' in inputs else None
            actions = actions.to(device)
            
            optimizer.zero_grad()
            predictions, new_hx = model(x_vis, x_aud=x_aud, x_thm=x_thm, hx=hx)            
            
            # 1. Mask the hidden state for individual sequence resets
            is_first = inputs['is_first'].to(device).float() # Shape: (batch_size, 1)

            if hx is not None:
                # Zero out hx for batch indices where is_first == 1
                mask = 1.0 - is_first 
                if isinstance(hx, (list, tuple)):
                    hx = [h * mask for h in hx]
                else:
                    hx = hx * mask

            # Use dynamic n_actions for the safety check
            if actions.shape[2] != n_actions:
                logger.error(f"CRITICAL: Data Mismatch! Found {actions.shape[2]} actions in data, but Brain expects {n_actions}.")
                return

            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
            
            # DETACH hx to prevent backpropagating through the entire history of the session
            # This implements the "Truncated" part of Truncated BPTT
            if isinstance(new_hx, (list, tuple)):
                hx = [h.detach() for h in new_hx]
            else:
                hx = new_hx.detach()

            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.5f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} complete. Average Loss: {avg_loss:.5f}")

    duration = time.time() - start_time
    logger.info(f"Training finished in {duration:.2f}s.")
    
    # Save the archive model using the dynamic prefix generator
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_prefix = generate_model_prefix(cfg, date_str)
    
    archive_path = get_unique_filename(model_dir, model_prefix, "pth")
    
    torch.save(model.state_dict(), archive_path)
    logger.info(f"Model archive saved to: {archive_path}")
    
    # Update the active model
    torch.save(model.state_dict(), str(active_model_path))
    logger.info(f"Active model updated at: {active_model_path}")