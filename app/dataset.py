"""
Data Module: The Memory Stream.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset # CHANGED from IterableDataset
import logging

logger = logging.getLogger(__name__)

class DoomStreamingDataset(Dataset):
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", augment=False, action_names=None):
        self.seq_len = seq_len
        self.augment = augment
        self.action_names = action_names or []
        self.samples = [] # Pre-allocated memory map
        
        # Build dynamic swap map for Mirror Augmentation
        self.swap_pairs = []
        if self.augment and self.action_names:
            self._build_swap_map()

        files = sorted(list(Path(data_dir).glob(file_pattern)))
        
        # 1. Load everything into RAM for random access shuffling
        for file_path in files:
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

                total_frames = len(frames)
                if total_frames < self.seq_len:
                    continue

                for i in range(total_frames - self.seq_len):
                    # Transpose immediately: (Seq, H, W, C) -> (Seq, C, H, W)
                    window_frames = np.transpose(frames[i : i + self.seq_len], (0, 3, 1, 2))
                    x = torch.from_numpy(window_frames).float()
                    y = torch.from_numpy(actions[i : i + self.seq_len]).float()

                    self.samples.append((x, y))

                    # Apply Augmentation directly to memory
                    if self.augment:
                        x_flip = torch.flip(x, [3])
                        y_flip = y.clone()
                        
                        for left_idx, right_idx in self.swap_pairs:
                            y_flip[:, left_idx] = y[:, right_idx]
                            y_flip[:, right_idx] = y[:, left_idx]
                            
                        self.samples.append((x_flip, y_flip))
                        
        logger.info(f"Dataset mapped to RAM: {len(self.samples)} total sequences available.")

    def _build_swap_map(self):
        try:
            self.swap_pairs.append((self.action_names.index("MOVE_LEFT"), self.action_names.index("MOVE_RIGHT")))
        except ValueError: pass
        
        try:
            self.swap_pairs.append((self.action_names.index("TURN_LEFT"), self.action_names.index("TURN_RIGHT")))
        except ValueError: pass

    # 2. Implement Map-Style required methods
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]