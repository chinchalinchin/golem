import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from pathlib import Path

from app.config import GolemConfig
from app.dataset import DoomStreamingDataset
from app.brain import DoomLiquidNet
from app.utils import resolve_path

logger = logging.getLogger(__name__)

def train_agent(cfg: GolemConfig):
    logger.info("Initializing Training Pipeline...")
    
    # 1. Setup Device (MPS for Mac M-chips, CUDA for Nvidia, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Device: Apple Metal (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using Device: NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Using Device: CPU")

    # 2. Prepare Data
    data_dir = resolve_path(cfg.data.output_dir)
    dataset = DoomStreamingDataset(data_dir, seq_len=cfg.training.sequence_length)
    
    # Batch size needs to be handled carefully with IterableDatasets.
    # The loader fetches 'batch_size' sequences.
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    
    # 3. Initialize Model
    # 3 Actions: Left, Right, Attack
    model = DoomLiquidNet(n_actions=3).to(device)
    
    # 4. Optimizer & Loss
    # We use BCEWithLogitsLoss because:
    #   - It handles Multi-Label classification (e.g., Moving Left AND Shooting simultaneously)
    #   - It is numerically more stable than using Sigmoid + MSE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # 5. The Loop
    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    
    model.train() # Set to training mode (enables Dropout, etc.)
    
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        batches = 0
        
        for batch_idx, (frames, actions) in enumerate(dataloader):
            # Move data to GPU/MPS
            frames = frames.to(device)   # (Batch, Time, Channels, Height, Width)
            actions = actions.to(device) # (Batch, Time, Actions)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward Pass (The Brain Thinks)
            # Output shape: (Batch, Time, Actions) - raw logits
            predictions = model(frames)
            
            # Calculate Loss (How wrong was it?)
            loss = criterion(predictions, actions)
            
            # Backward Pass (The Brain Learns)
            loss.backward()
            
            # Update Weights
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        # End of Epoch Stats
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} complete. Average Loss: {avg_loss:.4f}")

    # 6. Save the Brain
    save_path = resolve_path(cfg.training.model_save_path)
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training complete. Model saved to: {save_path}")