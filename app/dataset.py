"""
Data Module: The Memory Stream.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset

class DoomStreamingDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", augment=False, action_names=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.files = sorted(list(self.data_dir.glob(file_pattern)))
        self.augment = augment
        self.action_names = action_names or []
        
        # Build dynamic swap map for Mirror Augmentation
        self.swap_pairs = []
        if self.augment and self.action_names:
            self._build_swap_map()

    def _build_swap_map(self):
        try:
            self.swap_pairs.append((self.action_names.index("MOVE_LEFT"), self.action_names.index("MOVE_RIGHT")))
        except ValueError: pass
        
        try:
            self.swap_pairs.append((self.action_names.index("TURN_LEFT"), self.action_names.index("TURN_RIGHT")))
        except ValueError: pass

    def __iter__(self):
        for file_path in self.files:
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

            total_frames = len(frames)
            if total_frames < self.seq_len:
                continue

            for i in range(total_frames - self.seq_len):
                window_frames = np.transpose(frames[i : i + self.seq_len], (0, 3, 1, 2))
                x = torch.from_numpy(window_frames).float()
                y = torch.from_numpy(actions[i : i + self.seq_len]).float()

                yield x, y

                if self.augment:
                    x_flip = torch.flip(x, [3])
                    y_flip = y.clone()
                    
                    # Apply dynamic swaps
                    for left_idx, right_idx in self.swap_pairs:
                        y_flip[:, left_idx] = y[:, right_idx]
                        y_flip[:, right_idx] = y[:, left_idx]
                        
                    yield x_flip, y_flip