import torch.nn as nn
from ncps.torch import CfC

class DoomLiquidNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.liquid_rnn = CfC(
            input_size=12544, 
            units=64, 
            return_sequences=True 
        )
        
        self.output = nn.Linear(64, n_actions)

    def forward(self, x, hx=None):
        # x: (Batch, Time, C, H, W)
        batch, time, c, h, w = x.size()
        
        c_in = x.view(batch * time, c, h, w)
        c_out = self.conv(c_in)
        
        r_in = c_out.view(batch, time, -1)
        
        # Pass hidden state (hx) into the RNN and return the new state
        r_out, new_hx = self.liquid_rnn(r_in, hx)
        
        return self.output(r_out), new_hx