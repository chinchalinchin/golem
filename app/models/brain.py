"""
Brain Module: Neural Circuit Policy architecture for Golem.

This module defines the primary continuous-time neural network (LNN) used by the agent,
combining a dynamically scaling Convolutional Neural Network (CNN) visual cortex with
a Closed-form Continuous-time (CfC) liquid recurrent core.
"""

import torch
import torch.nn as nn
from ncps.torch import CfC

class DoomLiquidNet(nn.Module):
    r"""
    A continuous-time neural network for visual processing and temporal sequential decision-making.

    This network acts as the agent's brain. It processes raw pixel buffers through a 
    Convolutional Neural Network (Visual Cortex) to extract spatial features, which are 
    then fed into a Closed-form Continuous-time (CfC) recurrent network (Liquid Core). 
    The CfC core manages the agent's temporal state using differential equation approximations,
    allowing it to handle variable time-steps without requiring expensive ODE solvers.

    Args:
        n_actions (int): The number of output actions for the Motor Cortex head.
        cortical_depth (int, optional): The number of convolutional layers to generate. 
            Each layer halves the spatial dimensions and doubles the feature channels. 
            Default: ``2``.
        working_memory (int, optional): The number of hidden units in the CfC liquid core,
            representing the capacity of the agent's temporal memory. Default: ``64``.
        sensors: TODO
    """
    def __init__(self, n_actions, cortical_depth=2, working_memory=64, sensors=None):
        super().__init__()
        self.sensors = sensors
        
        # 1. Build the Visual Cortex (CNN) dynamically
        layers = []
        in_channels = 4 if self.sensors and getattr(self.sensors, 'depth', False) else 3
        out_channels = 32
        current_img_size = 64
        
        for i in range(cortical_depth):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2))
            layers.append(nn.ReLU())
            current_img_size = (current_img_size - 4) // 2 + 1
            in_channels = out_channels
            out_channels *= 2
            
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        
        flat_size = in_channels * (current_img_size ** 2)
        
        # 2. Build Auditory Cortex (Parallel 1D CNN)
        self.use_audio = self.sensors and getattr(self.sensors, 'audio', False)
        if self.use_audio:
            aud_layers = []
            a_in = 2 # Stereo channels
            a_out = 16
            for i in range(3):
                aud_layers.append(nn.Conv1d(a_in, a_out, kernel_size=8, stride=4))
                aud_layers.append(nn.ReLU())
                a_in = a_out
                a_out *= 2
            aud_layers.append(nn.AdaptiveAvgPool1d(1))
            aud_layers.append(nn.Flatten())
            self.audio_conv = nn.Sequential(*aud_layers)
            
            flat_size += a_in 
            
        # 3. Liquid Core
        self.liquid_rnn = CfC(
            input_size=flat_size, 
            units=working_memory, 
            return_sequences=True 
        )
        
        # 4. Motor Cortex Head
        self.output = nn.Linear(working_memory, n_actions)

    def forward(self, x_vis, x_aud=None, hx=None):
        r"""
        Performs a forward pass through the visual cortex and liquid core.

        Args:
            x_vis (Tensor): A batched sequence of visual frames of shape 
                :math:`(\text{Batch}, \text{Time}, C, H, W)`.
            x_aud (Tensor, optional): TODO
            hx (Tensor, optional): The previous hidden state of the liquid core of shape 
                :math:`(\text{Batch}, \text{working\_memory})`. Default: ``None``.

        Returns:
            tuple: A tuple containing:
                - Tensor: The unnormalized action logits of shape :math:`(\text{Batch}, \text{Time}, \text{n\_actions})`.
                - Tensor: The updated hidden state (working memory) for the next time-step.
        """
        batch, time, c, h, w = x_vis.size()
        
        c_in = x_vis.view(batch * time, c, h, w)
        features = self.conv(c_in)
        
        if self.use_audio and x_aud is not None:
            b, t, ac, a_s = x_aud.size()
            a_in = x_aud.view(b * t, ac, a_s)
            a_feat = self.audio_conv(a_in)
            features = torch.cat((features, a_feat), dim=1)
            
        r_in = features.view(batch, time, -1)
        r_out, new_hx = self.liquid_rnn(r_in, hx)
        
        return self.output(r_out), new_hx