"""
Training Module: Curriculum Learning.
Can train on a specific module's data or all available data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

from app.config import GolemConfig
from app.dataset import DoomStreamingDataset
from app.brain import DoomLiquidNet
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def train_agent(cfg: GolemConfig, module_name: str = None):
    """
    Trains the agent.
    
    Args:
        cfg: App config.
        module_name: If provided, only trains on files matching 'doom_training_<module>*.npz'.
                     If None, trains on ALL 'doom_training_*.npz' files.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 1. Filter Data Files
    data_dir = resolve_path(cfg.data.output_dir)
    
    # If module is specific, we filter the dataset class
    # Note: We need to update Dataset to accept a file pattern filter
    file_pattern = f"{cfg.data.filename_prefix}*.npz"
    if module_name:
        file_pattern = f"{cfg.data.filename_prefix}_{module_name}*.npz"
        logger.info(f"Training restricted to module: {module_name}")
    else:
        logger.info("Training on ALL available modules (Generalization Mode)")

    # 2. Initialize Dataset with Filter
    dataset = DoomStreamingDataset(
        data_dir, 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern
    )
    
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    
    # 3. Initialize Model with Superset Action Space
    n_actions = cfg.training.action_space_size
    logger.info(f"Initializing Brain with Action Space Size: {n_actions}")
    
    model = DoomLiquidNet(n_actions=n_actions).to(device)
    
    # Load existing weights if they exist (Continual Learning)
    save_path = resolve_path(cfg.training.model_save_path)
    if os.path.exists(save_path):
        logger.info(f"Loading existing brain from {save_path} for fine-tuning...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    model.train()
    
    # ... (Loop remains same) ...
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        batches = 0
        
        for batch_idx, (frames, actions) in enumerate(dataloader):
            frames = frames.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            predictions = model(frames)
            
            # Safety Check: Ensure action dimensions match
            # If old data (3 actions) is loaded into new model (8 actions), this crashes.
            if actions.shape[2] != n_actions:
                logger.error(f"Dimension Mismatch! Data has {actions.shape[2]} actions, Brain expects {n_actions}.")
                logger.error("Please delete old .npz files in 'data/' that were recorded with the old config.")
                return

            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} complete. Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training complete. Model saved to: {save_path}")