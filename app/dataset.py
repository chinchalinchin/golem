import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset

class DoomStreamingDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=32):
        """
        Args:
            data_dir: Path to the 'data' directory.
            seq_len: How many frames to feed the LNN at once (default 32).
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        
        # 1. Find all files
        # We sort them to ensure we process them in a deterministic order
        self.files = sorted(list(self.data_dir.glob("*.npz")))

    def __iter__(self):
        """
        This is where the magic happens. When the Training Loop asks for data,
        this function wakes up.
        """
        
        # Loop through every file in our data folder
        for file_path in self.files:
            
            # 2. Load the file into memory (It's okay to load one 50MB file at a time)
            with np.load(file_path) as data:
                frames = data['frames']  # Shape: (Total_Frames, 64, 64, 3)
                actions = data['actions'] # Shape: (Total_Frames, 3)

            # Safety check: If recording is too short, skip it
            total_frames = len(frames)
            if total_frames < self.seq_len:
                continue

            # 3. The Sliding Window
            # If we have 100 frames and seq_len is 32:
            # We can start a window at index 0, 1, ... up to 68.
            for i in range(total_frames - self.seq_len):
                
                # Slicing: Get frames i to i+32
                window_frames = frames[i : i + self.seq_len]
                window_actions = actions[i : i + self.seq_len]

                # 4. The Transformation (Crucial Step!)
                # PyTorch expects images as (Channels, Height, Width).
                # Currently they are (Time, Height, Width, Channels).
                # We need to move 'Channels' to the 2nd dimension.
                
                # Current: (32, 64, 64, 3)
                # Target:  (32, 3, 64, 64)
                window_frames = np.transpose(window_frames, (0, 3, 1, 2))

                # Convert to PyTorch Tensors (The native language of the Brain)
                x = torch.from_numpy(window_frames).float()
                y = torch.from_numpy(window_actions).float()

                # Hand the data to the trainer and pause
                yield x, y