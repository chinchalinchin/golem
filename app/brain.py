import torch
import torch.nn as nn
from ncps.torch import CfC

class DoomLiquidNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        # 1. Visual Cortex (CNN)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 2. Liquid Core (The "Brain")
        # Calculated Input: 64 channels * 14 * 14 = 12544
        self.liquid_rnn = CfC(input_size=12544, units=64, return_sequences=False)
        
        # 3. Motor Cortex
        self.output = nn.Linear(64, n_actions)

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        batch, time, c, h, w = x.size()
        
        # Merge Batch and Time for the CNN (it handles frames individually)
        c_in = x.view(batch * time, c, h, w)
        c_out = self.conv(c_in)
        
        # Unmerge for the RNN (it needs the Time sequence)
        r_in = c_out.view(batch, time, -1)
        
        # Run the Liquid Network
        r_out, _ = self.liquid_rnn(r_in)
        
        # Predict action
        return self.output(r_out)