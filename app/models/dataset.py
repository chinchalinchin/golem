"""
Data Module: The Memory Stream.

This module provides the data loading pipeline for Golem, converting raw gameplay
recordings (.npz files) into PyTorch-compatible datasets. It utilizes a sliding
window architecture to generate overlapping temporal sequences efficiently without 
duplicating flat arrays in memory.
"""

import torch
import torchaudio
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
    
    It supports multi-modal sensor fusion, dynamically yielding a dictionary of active 
    sensory tensors (visual, depth, audio, and thermal masks) alongside the target action vectors.
    It also supports dynamic horizontal mirror augmentation to double the effective
    dataset size while mitigating left/right turning bias.

    Args:
        data_dir (str or Path): The directory containing the ``.npz`` training files.
        seq_len (int, optional): The temporal length of the sequence window to slice. 
            Default: ``32``.
        file_pattern (str, optional): The glob pattern used to locate training files 
            within ``data_dir``. Default: ``"*.npz"``.
        augment (bool, optional): If ``True``, dynamically mirrors visual and thermal frames horizontally 
            and swaps corresponding left/right action labels. Default: ``False``.
        action_names (list of str, optional): The ordered list of string action names 
            (e.g., ``["MOVE_FORWARD", "TURN_LEFT", ...]``) used to calculate 
            which indices to swap during mirror augmentation. Default: ``None``.
        dsp_config (DSPConfig, optional): DSP tuning parameters for the Mel Spectrogram.
    """
    
    def __init__(self, data_dir, seq_len=32, file_pattern="*.npz", augment=False, action_names=None, dsp_config=None):
        self.seq_len = seq_len
        self.augment = augment
        self.action_names = action_names or []
        self.dsp_config = dsp_config
        
        # Memory stores
        self.video_arrays = []
        self.action_arrays = []
        self.depth_arrays = []
        self.audio_arrays = []
        self.thermal_arrays = []
        
        self.has_depth = False
        self.has_audio = False
        self.has_thermal = False
        
        self.index_map = [] 
        
        self.swap_pairs = []
        if self.augment and self.action_names:
            self._build_swap_map()

        files = sorted(list(Path(data_dir).glob(file_pattern)))
        
        for file_idx, file_path in enumerate(files):
            with np.load(file_path) as data:
                frames = data['frames']
                actions = data['actions']

                self.video_arrays.append(frames)
                self.action_arrays.append(actions)
                
                if 'depths' in data:
                    self.has_depth = True
                    self.depth_arrays.append(data['depths'])
                if 'audios' in data:
                    self.has_audio = True
                    self.audio_arrays.append(data['audios'])
                if 'thermals' in data:
                    self.has_thermal = True
                    self.thermal_arrays.append(data['thermals'])

                total_frames = len(frames)
                if total_frames < self.seq_len:
                    continue

                for start_idx in range(total_frames - self.seq_len):
                    self.index_map.append((file_idx, start_idx, False))
                    if self.augment:
                        self.index_map.append((file_idx, start_idx, True))
                        
        # Setup DSP transforms if audio is present
        self.mel_transform = None
        self.amp_to_db = None
        if self.has_audio and self.dsp_config:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.dsp_config.sample_rate,
                n_fft=self.dsp_config.n_fft,
                hop_length=self.dsp_config.hop_length,
                n_mels=self.dsp_config.n_mels
            )
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        logger.info(f"Dataset mapped to RAM using pointers: {len(self.index_map)} sequences available. Modalities: [Visual: True, Depth: {self.has_depth}, Audio: {self.has_audio}, Thermal: {self.has_thermal}]")

    def _build_swap_map(self):
        r"""
        Constructs a mapping of action indices that must be swapped when applying 
        horizontal mirror augmentation.

        This ensures that when a spatial tensor (visual or thermal) is visually flipped, an action like 
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

        The retrieved visual frames are dynamically transposed from the storage shape of 
        :math:`(H, W, C)` to the PyTorch convolutional shape of :math:`(C, H, W)`.
        If the index maps to an augmented sequence, the visual and thermal tensors are flipped 
        horizontally and lateral actions are swapped. For audio, the waveforms 
        are converted to Mel Spectrograms and spatial auditory channels are swapped.

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
        file_idx, start_idx, is_mirrored = self.index_map[idx]
        
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
            
            if self.mel_transform and self.amp_to_db:
                x_aud = self.mel_transform(x_aud)
                x_aud = self.amp_to_db(x_aud)
                
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
            return inputs, y_flip
            
        inputs = {'visual': x_vis}
        if self.has_audio:
            inputs['audio'] = x_aud
        if self.has_thermal:
            inputs['thermal'] = x_thm
            
        return inputs, y