"""
Training Module: Curriculum Learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import time

from app.config import GolemConfig
from app.dataset import DoomStreamingDataset
from app.brain import DoomLiquidNet
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def train_agent(cfg: GolemConfig, module_name: str = None):
    # 1. Hardware Acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Apple Metal (MPS) acceleration detected and enabled.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("🚀 CUDA acceleration detected and enabled.")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ No GPU detected. Training will be slow on CPU.")
    
    # 2. Filter Data Files
    data_dir = resolve_path(cfg.data.output_dir)
    
    if module_name and module_name.lower() == "all":
        module_name = None

    file_pattern = f"{cfg.data.filename_prefix}*.npz"
    if module_name:
        file_pattern = f"{cfg.data.filename_prefix}_{module_name}*.npz"
        logger.info(f"Training restricted to module: {module_name}")
    else:
        logger.info("Training on ALL available modules (Generalization Mode)")

    # 3. Initialize Dataset (WITH AUGMENTATION)
    # -----------------------------------------
    do_augment = cfg.training.augmentation.mirror
    if do_augment:
        logger.info("🔮 Mirror Augmentation ENABLED. Dataset size will effectively double.")
        
    dataset = DoomStreamingDataset(
        data_dir, 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern,
        augment=do_augment  # Pass the flag
    )
    
    if not dataset.files:
        logger.error(f"No training data found matching pattern: {file_pattern}")
        return

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    
    # 4. Initialize Model
    n_actions = cfg.training.action_space_size
    model = DoomLiquidNet(n_actions=n_actions).to(device)
    
    save_path = resolve_path(cfg.training.model_save_path)
    if os.path.exists(save_path):
        logger.info(f"Loading existing brain from {save_path} for fine-tuning...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    
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
            predictions = model(frames)
            
            if actions.shape[2] != n_actions:
                logger.error(f"CRITICAL: Data Mismatch! Found {actions.shape[2]} actions, Brain expects {n_actions}.")
                return

            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"✅ Epoch {epoch+1}/{cfg.training.epochs} complete. Average Loss: {avg_loss:.4f}")

    duration = time.time() - start_time
    logger.info(f"Training finished in {duration:.2f}s.")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to: {save_path}")