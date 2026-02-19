"""
Data Module: The Memory Stream.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset

class DoomStreamingDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", augment=False):
        """
        Args:
            data_dir: Path to data.
            seq_len: Window size.
            file_pattern: Glob pattern to filter files.
            augment: If True, applies data augmentation (Mirroring).
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.files = sorted(list(self.data_dir.glob(file_pattern)))
        self.augment = augment

    def __iter__(self):
        for file_path in self.files:
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

            total_frames = len(frames)
            if total_frames < self.seq_len:
                continue

            for i in range(total_frames - self.seq_len):
                window_frames = frames[i : i + self.seq_len]
                window_actions = actions[i : i + self.seq_len]

                # (T, H, W, C) -> (T, C, H, W)
                window_frames = np.transpose(window_frames, (0, 3, 1, 2))

                x = torch.from_numpy(window_frames).float()
                y = torch.from_numpy(window_actions).float()

                # Yield Original
                yield x, y

                # Yield Mirrored (if enabled)
                if self.augment:
                    # 1. Flip Image Horizontally (Axis 3 is Width)
                    x_flip = torch.flip(x, [3])
                    
                    # 2. Swap Actions
                    # [0:Fwd, 1:Back, 2:MoveL, 3:MoveR, 4:TurnL, 5:TurnR, 6:Atk, 7:Use]
                    y_flip = y.clone()
                    
                    # Swap Move Left (2) <-> Move Right (3)
                    y_flip[:, 2] = y[:, 3]
                    y_flip[:, 3] = y[:, 2]
                    
                    # Swap Turn Left (4) <-> Turn Right (5)
                    y_flip[:, 4] = y[:, 5]
                    y_flip[:, 5] = y[:, 4]
                    
                    yield x_flip, y_flip