# Standard Libraries
import logging
from pathlib import Path
from typing import List

# External Libraries
import numpy as np
import cv2
import torch

logger = logging.getLogger(__name__)


MODEL_ARCHIVE_TEMPLATE = "{date}.c-{c}.w-{w}.v-{v}.d-{d}.a-{a}.t-{t}.sr-{sr}.nf-{nf}.hl-{hl}.nm-{nm}"


def get_latest_parameters(archives: List[Path]) -> dict:
    """
    Find the latest parameters of the trained model from the filename.
    """
    params = {}
    if archives:
        latest_archive = sorted(archives, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        try:
            parts = latest_archive.name.split('.')
            for part in parts:
                if part.startswith('c-'): params['c'] = int(part[2:])
                elif part.startswith('w-'): params['w'] = int(part[2:])
                elif part.startswith('v-'): params['v'] = bool(int(part[2:]))
                elif part.startswith('d-'): params['d'] = bool(int(part[2:]))
                elif part.startswith('a-'): params['a'] = bool(int(part[2:]))
                elif part.startswith('t-'): params['t'] = bool(int(part[2:]))
                elif part.startswith('sr-'): params['sr'] = int(part[3:])
                elif part.startswith('nf-'): params['nf'] = int(part[3:])
                elif part.startswith('hl-'): params['hl'] = int(part[3:])
                elif part.startswith('nm-'): params['nm'] = int(part[3:])
            logger.info(f"Discovered brain architecture from {latest_archive.name}: {params}")
        except Exception as e:
            logger.warning(f"Failed to parse architecture from {latest_archive.name}: {e}")
    return params


def apply_latest_parameters(cfg, archives: List[Path]) -> None:
    """Helper to parse and apply filename parameters to the active config."""
    params = get_latest_parameters(archives)
    if params:
        cfg.brain.cortical_depth = params.get('c', cfg.brain.cortical_depth)
        cfg.brain.working_memory = params.get('w', cfg.brain.working_memory)
        cfg.brain.sensors.visual = params.get('v', cfg.brain.sensors.visual)
        cfg.brain.sensors.depth = params.get('d', cfg.brain.sensors.depth)
        cfg.brain.sensors.audio = params.get('a', cfg.brain.sensors.audio)
        cfg.brain.sensors.thermal = params.get('t', cfg.brain.sensors.thermal)
        cfg.brain.dsp.sample_rate = params.get('sr', cfg.brain.dsp.sample_rate)
        cfg.brain.dsp.n_fft = params.get('nf', cfg.brain.dsp.n_fft)
        cfg.brain.dsp.hop_length = params.get('hl', cfg.brain.dsp.hop_length)
        cfg.brain.dsp.n_mels = params.get('nm', cfg.brain.dsp.n_mels)


def normalize_audio_buffer(raw_audio: np.ndarray) -> np.ndarray:
    """Applies zero-mean, unit-variance normalization to a raw engine audio buffer."""
    mean = np.mean(raw_audio, axis=-1, keepdims=True)
    std = np.std(raw_audio, axis=-1, keepdims=True) + 1e-8
    return (raw_audio - mean) / std


class SensoryExtractor:
    """
    Centralized utility class for extracting, normalizing, and formatting 
    ViZDoom phenomenological buffers into numpy arrays and PyTorch tensors.
    """
    
    @staticmethod
    def get_numpy_state(state, sensors_cfg) -> dict:
        """Extracts and normalizes raw game state buffers into a dictionary."""
        data = {}
        
        # 1. Visual (RGB)
        if state.screen_buffer is not None:
            data['visual'] = cv2.resize(state.screen_buffer.transpose(1, 2, 0), (64, 64)) / 255.0
            
        # 2. Depth
        if getattr(sensors_cfg, 'depth', False) and state.depth_buffer is not None:
            data['depth'] = cv2.resize(state.depth_buffer, (64, 64)) / 255.0
            
        # 3. Audio (Normalized Raw Waveform)
        if getattr(sensors_cfg, 'audio', False) and state.audio_buffer is not None:
            raw_audio = state.audio_buffer
            mean = np.mean(raw_audio, axis=-1, keepdims=True)
            std = np.std(raw_audio, axis=-1, keepdims=True) + 1e-8
            data['audio'] = (raw_audio - mean) / std
            
        # 4. Thermal
        if getattr(sensors_cfg, 'thermal', False) and state.labels_buffer is not None:
            binary_mask = (state.labels_buffer > 0).astype(np.float32)
            data['thermal'] = cv2.resize(binary_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
            
        return data

    @staticmethod
    def to_tensors(numpy_state: dict, device: torch.device) -> dict:
        """Converts normalized numpy dictionary into PyTorch tensors for model inference."""
        tensors = {}
        
        # Visual & Depth Concatenation
        if 'visual' in numpy_state:
            x_vis_np = numpy_state['visual']
            if 'depth' in numpy_state:
                depth_expanded = np.expand_dims(numpy_state['depth'], axis=2)
                x_vis_np = np.concatenate((x_vis_np, depth_expanded), axis=2)
            tensors['visual'] = torch.from_numpy(np.transpose(x_vis_np, (2, 0, 1))).float().unsqueeze(0).unsqueeze(0).to(device)
            
        # Audio (Now passes the raw 1D waveform directly to the model)
        if 'audio' in numpy_state:
            tensors['audio'] = torch.from_numpy(numpy_state['audio']).float().unsqueeze(0).unsqueeze(0).to(device)
            
        # Thermal Mask
        if 'thermal' in numpy_state:
            tensors['thermal'] = torch.from_numpy(numpy_state['thermal']).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        
        return tensors