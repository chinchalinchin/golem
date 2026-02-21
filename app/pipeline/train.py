"""
Training Module: Behavioral Cloning Loop.

This module handles the supervised learning pipeline for the Golem agent. 
It implements a behavioral cloning loop that maps visual sequence inputs 
(screen buffers) to expert action vectors using Binary Cross-Entropy loss, 
effectively teaching the agent to mimic human gameplay demonstrations.
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

# Application Libraries
from app.models.config import GolemConfig
from app.models.dataset import DoomStreamingDataset
from app.models.brain import DoomLiquidNet
from app.utils import resolve_path, get_unique_filename, register_command, get_latest_parameters

logger = logging.getLogger(__name__)

@register_command("train")
def train(cfg: GolemConfig, module_name: str = None):
    r"""
    Trains the Liquid Neural Network using captured expert demonstrations.

    This function orchestrates the dataset streaming and the model's training loop. 
    It dynamically selects the best available hardware accelerator (CUDA, MPS, or CPU), 
    initializes the dataset with optional mirror augmentation, and optimizes the 
    network weights using the Adam optimizer.

    If an active model already exists for the current profile (e.g., ``fluid``), 
    it loads the existing weights to perform continuous fine-tuning. Upon completion, 
    the updated model is saved to both a timestamped archive and the active profile slot.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): The specific data module to train against 
            (e.g., "combat", "navigation"). If ``"all"`` or ``None``, it trains 
            across all available data for the active profile (Generalization Mode). 
            Default: ``None``.
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
    data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    prefix_clean = cfg.data.prefix.rstrip('_')
    
    if module_name and module_name.lower() != "all":
        file_pattern = f"{prefix_clean}_{module_name}*.npz"
        logger.info(f"Training restricted to module: {module_name}")
    else:
        file_pattern = f"{prefix_clean}_*.npz"        
        logger.info("Training on ALL available modules (Generalization Mode)")

    dataset = DoomStreamingDataset(
        str(data_dir), 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern,
        augment=cfg.training.augmentation.mirror,
        action_names=cfg.training.action_names 
    )
    
    if len(dataset) == 0:
        logger.error(f"No training data found matching pattern: {file_pattern} in {data_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)    
    
    # 1. Base Defaults
    cortical_depth = cfg.brain.cortical_depth
    working_memory = cfg.brain.working_memory
    n_actions = cfg.training.action_space_size

    # 2. Discover architecture and dimensions if resuming training
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = data_dir / "golem.pth"
    state_dict = None

    if active_model_path.exists():
        logger.info(f"Discovering existing brain architecture from {active_model_path} for fine-tuning...")
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]

        archives = list(model_dir.glob("*.pth"))
        params = get_latest_parameters(archives)
        if params:
            cortical_depth, working_memory = params

    # 3. Initialize dynamic model
    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cortical_depth,
        working_memory=working_memory
    ).to(device)    
    
    if state_dict:
        model.load_state_dict(state_dict)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    model.train()
    
    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        total_loss = 0
        batches = 0
        
        for batch_idx, (frames, actions) in enumerate(dataloader):
            frames = frames.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(frames) 
            
            # Use dynamic n_actions for the safety check
            if actions.shape[2] != n_actions:
                logger.error(f"CRITICAL: Data Mismatch! Found {actions.shape[2]} actions in data, but Brain expects {n_actions}.")
                return

            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} complete. Average Loss: {avg_loss:.4f}")

    duration = time.time() - start_time
    logger.info(f"Training finished in {duration:.2f}s.")
    
    # Save the archive model
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_prefix = f"{date_str}.c-{cortical_depth}.w-{working_memory}"
    archive_path = get_unique_filename(model_dir, model_prefix, "pth")
    
    torch.save(model.state_dict(), archive_path)
    logger.info(f"Model archive saved to: {archive_path}")
    
    # Update the active model
    torch.save(model.state_dict(), str(active_model_path))
    logger.info(f"Active model updated at: {active_model_path}")