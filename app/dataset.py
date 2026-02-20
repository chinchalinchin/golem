"""
Data Module: The Memory Stream.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class DoomStreamingDataset(Dataset):
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", augment=False, action_names=None):
        self.seq_len = seq_len
        self.augment = augment
        self.action_names = action_names or []
        
        # Memory stores
        self.video_arrays = []
        self.action_arrays = []
        self.index_map = [] # Pointers: (file_index, frame_start_index, is_mirrored)
        
        self.swap_pairs = []
        if self.augment and self.action_names:
            self._build_swap_map()

        files = sorted(list(Path(data_dir).glob(file_pattern)))
        
        # 1. Load flat arrays to RAM (No duplication)
        for file_idx, file_path in enumerate(files):
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

                self.video_arrays.append(frames)
                self.action_arrays.append(actions)

                total_frames = len(frames)
                if total_frames < self.seq_len:
                    continue

                # 2. Build the lightweight pointers
                for start_idx in range(total_frames - self.seq_len):
                    self.index_map.append((file_idx, start_idx, False))
                    
                    if self.augment:
                        self.index_map.append((file_idx, start_idx, True))
                        
        logger.info(f"Dataset mapped to RAM using pointers: {len(self.index_map)} sequences available.")

    def _build_swap_map(self):
        try:
            self.swap_pairs.append((self.action_names.index("MOVE_LEFT"), self.action_names.index("MOVE_RIGHT")))
        except ValueError: pass
        
        try:
            self.swap_pairs.append((self.action_names.index("TURN_LEFT"), self.action_names.index("TURN_RIGHT")))
        except ValueError: pass

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 3. Slice the 32 frames on-the-fly
        file_idx, start_idx, is_mirrored = self.index_map[idx]
        
        window_frames = self.video_arrays[file_idx][start_idx : start_idx + self.seq_len]
        window_actions = self.action_arrays[file_idx][start_idx : start_idx + self.seq_len]

        # (Seq, H, W, C) -> (Seq, C, H, W)
        window_frames = np.transpose(window_frames, (0, 3, 1, 2))
        
        x = torch.from_numpy(window_frames).float()
        y = torch.from_numpy(window_actions).float()

        if is_mirrored:
            x = torch.flip(x, [3])
            y_flip = y.clone()
            
            for left_idx, right_idx in self.swap_pairs:
                y_flip[:, left_idx] = y[:, right_idx]
                y_flip[:, right_idx] = y[:, left_idx]
                
            return x, y_flip
            
        return x, y