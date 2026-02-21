"""
Data Module: The Memory Stream.

This module provides the data loading pipeline for Golem, converting raw gameplay
recordings (.npz files) into PyTorch-compatible datasets. It utilizes a sliding
window architecture to generate overlapping temporal sequences efficiently without 
duplicating flat arrays in memory.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class DoomStreamingDataset(Dataset):
    r"""
    A PyTorch Dataset for loading and streaming DOOM gameplay sequences.

    This class loads raw frame and action arrays from compressed ``.npz`` files 
    into memory. Rather than copying the data to create individual sequence 
    tensors, it builds a lightweight pointer map. During training, it slices 
    continuous arrays into overlapping sequences of length ``seq_len`` on-the-fly.
    
    It also supports dynamic horizontal mirror augmentation to double the effective
    dataset size while mitigating left/right turning bias.

    Args:
        data_dir (str or Path): The directory containing the ``.npz`` training files.
        seq_len (int, optional): The temporal length of the sequence window to slice. 
            Default: ``32``.
        file_pattern (str, optional): The glob pattern used to locate training files 
            within ``data_dir``. Default: ``"*.npz"``.
        augment (bool, optional): If ``True``, dynamically mirrors frames horizontally 
            and swaps corresponding left/right action labels. Default: ``False``.
        action_names (list of str, optional): The ordered list of string action names 
            (e.g., ``["MOVE_FORWARD", "TURN_LEFT", ...]``) used to calculate 
            which indices to swap during mirror augmentation. Default: ``None``.
    """
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
        r"""
        Constructs a mapping of action indices that must be swapped when applying 
        horizontal mirror augmentation.

        This ensures that when a frame is visually flipped, an action like 
        ``TURN_LEFT`` correctly transforms into ``TURN_RIGHT`` in the target vector.
        """
        try:
            self.swap_pairs.append((self.action_names.index("MOVE_LEFT"), self.action_names.index("MOVE_RIGHT")))
        except ValueError: pass
        
        try:
            self.swap_pairs.append((self.action_names.index("TURN_LEFT"), self.action_names.index("TURN_RIGHT")))
        except ValueError: pass

    def __len__(self):
        r"""
        Returns the total number of sliding window sequences available.

        Returns:
            int: The total sequence count, including augmented sequences if enabled.
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        r"""
        Retrieves a temporal sequence of frames and corresponding actions by index.

        The retrieved frames are dynamically transposed from the storage shape of 
        :math:`(H, W, C)` to the PyTorch convolutional shape of :math:`(C, H, W)`.
        If the index maps to an augmented sequence, the visual tensor is flipped 
        horizontally and lateral actions are swapped.

        Args:
            idx (int): The index of the sequence pointer in the internal map.

        Returns:
            tuple: A tuple containing:
                - Tensor: A sequence of visual frames of shape :math:`(\text{seq\_len}, C, H, W)`.
                - Tensor: A sequence of action vectors of shape :math:`(\text{seq\_len}, \text{n\_actions})`.
        """
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