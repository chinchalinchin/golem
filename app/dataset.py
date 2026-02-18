"""
Data Module: The Memory Stream.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset

class DoomStreamingDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz"):
        """
        Args:
            data_dir: Path to data.
            seq_len: Window size.
            file_pattern: Glob pattern to filter files (e.g. '*_combat*.npz').
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        # Updated to use the pattern
        self.files = sorted(list(self.data_dir.glob(file_pattern)))

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

                yield x, y