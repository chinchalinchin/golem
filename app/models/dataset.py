"""
Data Module: The Memory Stream.

This module provides the data loading pipeline for Golem, converting raw gameplay
recordings (.npz files) into PyTorch-compatible datasets. It utilizes a sliding
window architecture to generate overlapping temporal sequences efficiently without 
duplicating flat arrays in memory.
"""

import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

class DoomStreamingDataset(Dataset):
    r"""
    A PyTorch Dataset for loading and streaming DOOM gameplay sequences.

    This class loads raw frame and action arrays from compressed ``.npz`` files into memory. Rather than copying the data to create individual sequence tensors, it builds a lightweight pointer map. During training, it slices continuous arrays into overlapping sequences of length ``seq_len`` on-the-fly.
    
    It supports multi-modal sensor fusion, dynamically yielding a dictionary of active sensory tensors (visual, depth, audio, and thermal masks) alongside the target action vectors. It also supports dynamic horizontal mirror augmentation to double the effective dataset size while mitigating left/right turning bias.

    Args:
        data_dir (str or Path): The directory containing the ``.npz`` training files.
        seq_len (int, optional): The temporal length of the sequence window to slice. Default: ``32``.
        file_pattern (str, optional): The glob pattern used to locate training files within ``data_dir``. Default: ``"*.npz"``.
        augment (bool, optional): If ``True``, dynamically mirrors visual and thermal frames horizontally and swaps corresponding left/right action labels. Default: ``False``.
        action_names (list of str, optional): The ordered list of string action names (e.g., ``["MOVE_FORWARD", "TURN_LEFT", ...]``) used to calculate which indices to swap during mirror augmentation. Default: ``None``.
        dsp_config (DSPConfig, optional): DSP tuning parameters for the Mel Spectrogram.
        sensors: TODO
    """
    
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", 
                    augment=False, action_names=None, dsp_config=None, sensors=None):
        self.seq_len = seq_len
        self.augment = augment
        self.action_names = action_names or []
        self.dsp_config = dsp_config
        self.sensors = sensors
        
        # Memory stores
        self.video_arrays = []
        self.action_arrays = []
        self.depth_arrays = []
        self.audio_arrays = []
        self.thermal_arrays = []
        
        # Enforce configuration-driven modalities instead of dynamic inference
        self.has_depth = getattr(self.sensors, 'depth', False) if self.sensors else False
        self.has_audio = getattr(self.sensors, 'audio', False) if self.sensors else False
        self.has_thermal = getattr(self.sensors, 'thermal', False) if self.sensors else False

        self.index_map = [] 
        self.base_episodes = []      # <-- NEW: Stores lists of contiguous base indices
        self.recovery_episodes = []  # <-- NEW: Stores lists of contiguous recovery indices
        
        self.swap_pairs = []
        if self.augment and self.action_names:
            self._build_swap_map()

        # Handle data_dir as a single string, Path, or a list of them
        if isinstance(data_dir, (str, Path)):
            search_dirs = [Path(data_dir)]
        else:
            search_dirs = [Path(d) for d in data_dir]

        files = []
        for d in search_dirs:
            if d.exists():
                files.extend(list(d.glob(file_pattern)))
        files = sorted(files)
        
        for file_idx, file_path in enumerate(files):
            is_recovery = "recovery" in str(file_path).lower() # <-- Identify origin
            
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

                self.video_arrays.append(frames)
                self.action_arrays.append(actions)
                
                if self.has_depth:
                    if 'depths' in data:
                        self.depth_arrays.append(data['depths'])
                    else:
                        raise ValueError(f"Config requires 'depth', but {file_path.name} is missing 'depths' array.")
                        
                if self.has_audio:
                    if 'audios' in data:
                        self.audio_arrays.append(data['audios'])
                    else:
                        raise ValueError(f"Config requires 'audio', but {file_path.name} is missing 'audios' array.")
                        
                if self.has_thermal:
                    if 'thermals' in data:
                        self.thermal_arrays.append(data['thermals'])
                    else:
                        raise ValueError(f"Config requires 'thermal', but {file_path.name} is missing 'thermals' array.")
                    
                total_frames = len(frames)
                if total_frames < self.seq_len:
                    continue

                episode_indices_normal = []
                episode_indices_mirrored = []

                # Build pointers and group them into contiguous episodes
                for start_idx in range(0, total_frames - self.seq_len + 1, self.seq_len):
                    is_first = (start_idx == 0)
                    global_idx = len(self.index_map)
                    
                    self.index_map.append({
                        'file_idx': file_idx, 
                        'start_idx': start_idx, 
                        'is_mirrored': False, 
                        'is_first': is_first
                    })
                    episode_indices_normal.append(global_idx)
                    
                    if self.augment:
                        global_idx_aug = len(self.index_map)
                        self.index_map.append({
                            'file_idx': file_idx, 
                            'start_idx': start_idx, 
                            'is_mirrored': True, 
                            'is_first': is_first
                        })
                        episode_indices_mirrored.append(global_idx_aug)
                        
                # Register completed episodes to their respective pools
                if episode_indices_normal:
                    if is_recovery: self.recovery_episodes.append(episode_indices_normal)
                    else: self.base_episodes.append(episode_indices_normal)
                    
                if self.augment and episode_indices_mirrored:
                    if is_recovery: self.recovery_episodes.append(episode_indices_mirrored)
                    else: self.base_episodes.append(episode_indices_mirrored)

        logger.info(f"Dataset mapped to RAM: {len(self.base_episodes)} base episodes, {len(self.recovery_episodes)} recovery episodes. Modalities: [Visual: True, Depth: {self.has_depth}, Audio: {self.has_audio}, Thermal: {self.has_thermal}]")

    def _build_swap_map(self):
        r"""
        Constructs a mapping of action indices that must be swapped when applying horizontal mirror augmentation.

        This ensures that when a spatial tensor (visual or thermal) is visually flipped, an action like ``TURN_LEFT`` correctly transforms into ``TURN_RIGHT`` in the target vector.
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

        The retrieved visual frames are dynamically transposed from the storage shape of :math:`(H, W, C)` to the PyTorch convolutional shape of :math:`(C, H, W)`.
        
        If the index maps to an augmented sequence, the visual and thermal tensors are flipped horizontally and lateral actions are swapped. For audio, the waveforms are converted to Mel Spectrograms and spatial auditory channels are swapped.

        Args:
            idx (int): The index of the sequence pointer in the internal map.

        Returns:
            tuple: A tuple containing:
                - dict: A dictionary of active sensory inputs:
                    - ``'visual'`` (Tensor): Visual frames of shape :math:`(\text{seq\_len}, C, 64, 64)`.
                    - ``'audio'`` (Tensor, optional): Mel spectrograms of shape :math:`(\text{seq\_len}, 2, H_{mels}, W_{time})`.
                    - ``'thermal'`` (Tensor, optional): Binary thermal masks of shape :math:`(\text{seq\_len}, 1, 64, 64)`.
                - Tensor: A sequence of action vectors of shape :math:`(\text{seq\_len}, \text{n\_actions})`.
        """
        meta = self.index_map[idx]
        file_idx, start_idx, is_mirrored, is_first = meta['file_idx'], meta['start_idx'], meta['is_mirrored'], meta['is_first']
        
        window_frames = self.video_arrays[file_idx][start_idx : start_idx + self.seq_len]
        window_actions = self.action_arrays[file_idx][start_idx : start_idx + self.seq_len]

        window_frames = np.transpose(window_frames, (0, 3, 1, 2))
        x_vis = torch.from_numpy(window_frames).float()
        y = torch.from_numpy(window_actions).float()

        if self.has_depth:
            window_depths = self.depth_arrays[file_idx][start_idx : start_idx + self.seq_len]
            window_depths = np.expand_dims(window_depths, axis=1)
            x_depth = torch.from_numpy(window_depths).float()
            x_vis = torch.cat((x_vis, x_depth), dim=1) 

        x_aud = None
        if self.has_audio:
            window_audios = self.audio_arrays[file_idx][start_idx : start_idx + self.seq_len]
            x_aud = torch.from_numpy(window_audios).float()
                
        x_thm = None
        if self.has_thermal:
            window_thermals = self.thermal_arrays[file_idx][start_idx : start_idx + self.seq_len]
            window_thermals = np.expand_dims(window_thermals, axis=1)
            x_thm = torch.from_numpy(window_thermals).float()

        if is_mirrored:
            x_vis = torch.flip(x_vis, [3])
            if self.has_audio:
                # Flip channel 1 (stereo channels) for spatial auditory swapping
                x_aud = torch.flip(x_aud, [1])
            if self.has_thermal:
                x_thm = torch.flip(x_thm, [3])
                
            y_flip = y.clone()
            for left_idx, right_idx in self.swap_pairs:
                y_flip[:, left_idx] = y[:, right_idx]
                y_flip[:, right_idx] = y[:, left_idx]
            
            inputs = {'visual': x_vis}
            if self.has_audio:
                inputs['audio'] = x_aud
            if self.has_thermal:
                inputs['thermal'] = x_thm
                
            inputs['is_first'] = torch.tensor([is_first], dtype=torch.bool)
            return inputs, y_flip
            
        inputs = {'visual': x_vis}
        if self.has_audio:
            inputs['audio'] = x_aud
        if self.has_thermal:
            inputs['thermal'] = x_thm

        inputs['is_first'] = torch.tensor([is_first], dtype=torch.bool)
        return inputs, y
        
class StatefulStratifiedBatchSampler(Sampler):
    """
    Custom Sampler designed to maintain continuous temporal streams across 
    batch dimensions for Stateful Backpropagation Through Time (BPTT).
    
    Dynamically mixes base expert trajectories with recovery (DAgger) trajectories 
    to prevent catastrophic forgetting.
    """
    def __init__(self, base_episodes, recovery_episodes, batch_size, recovery_ratio=0.25):
        self.base_episodes = base_episodes
        self.recovery_episodes = recovery_episodes
        self.batch_size = batch_size
        
        # Calculate stream allocations
        self.n_recovery = int(batch_size * recovery_ratio) if recovery_episodes else 0
        self.n_base = batch_size - self.n_recovery
        
        if self.n_base <= 0:
            raise ValueError("Batch size is too small or recovery ratio is too high to maintain base streams.")

    def __iter__(self):
        # 1. Initialize shuffled pools for the epoch
        base_pool = list(self.base_episodes)
        rec_pool = list(self.recovery_episodes)
        random.shuffle(base_pool)
        random.shuffle(rec_pool)
        
        active_streams = []
        
        # 2. Allocate initial temporal streams
        for i in range(self.batch_size):
            is_rec = (i < self.n_recovery)
            pool = rec_pool if is_rec else base_pool
            original = self.recovery_episodes if is_rec else self.base_episodes
            
            if not pool and original:  # Refill immediately if pool is unexpectedly small
                pool.extend(original)
                random.shuffle(pool)
                
            if pool:
                active_streams.append(iter(pool.pop()))
            else:
                active_streams.append(None)
                
        # 3. Yield continuous batches
        while True:
            batch = []
            for i in range(self.batch_size):
                is_rec = (i < self.n_recovery)
                pool = rec_pool if is_rec else base_pool
                original = self.recovery_episodes if is_rec else self.base_episodes
                
                stream = active_streams[i]
                if stream is None:
                    continue
                    
                try:
                    idx = next(stream)
                    batch.append(idx)
                except StopIteration:
                    # Stream exhausted. Fetch a new episode.
                    if not pool and original:
                        # Endlessly recycle recovery data. 
                        # Exhausting the base pool signals the end of the epoch.
                        if is_rec:
                            pool.extend(original)
                            random.shuffle(pool)
                            
                    if pool:
                        new_stream = iter(pool.pop())
                        active_streams[i] = new_stream
                        batch.append(next(new_stream))
                    else:
                        active_streams[i] = None
                        
            # Drop the last batch if it cannot fully populate the parallel streams
            if len(batch) == self.batch_size:
                yield batch
            else:
                break

    def __len__(self):
        if self.n_base == 0:
            return 0
        total_base_seqs = sum(len(ep) for ep in self.base_episodes)
        return total_base_seqs // self.n_base