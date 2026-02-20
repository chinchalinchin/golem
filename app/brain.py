import torch.nn as nn
from ncps.torch import CfC

class DoomLiquidNet(nn.Module):
    def __init__(self, n_actions, cortical_depth=2, working_memory=64):
        super().__init__()
        
        # 1. Build the Visual Cortex (CNN) dynamically
        layers = []
        in_channels = 3
        out_channels = 32
        current_img_size = 64
        
        for i in range(cortical_depth):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2))
            layers.append(nn.ReLU())
            
            # Calculate the new image dimension: (W - F) / S + 1
            current_img_size = (current_img_size - 4) // 2 + 1
            
            in_channels = out_channels
            out_channels *= 2 # Double the feature maps each layer
            
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        
        # Calculate the exact size of the flattened tensor
        flat_size = in_channels * (current_img_size ** 2)
        
        # 2. Build the Liquid Core with dynamic working memory
        self.liquid_rnn = CfC(
            input_size=flat_size, 
            units=working_memory, 
            return_sequences=True 
        )
        
        # 3. Motor Cortex Head
        self.output = nn.Linear(working_memory, n_actions)

    def forward(self, x, hx=None):
        # x: (Batch, Time, C, H, W)
        batch, time, c, h, w = x.size()
        
        c_in = x.view(batch * time, c, h, w)
        c_out = self.conv(c_in)
        
        r_in = c_out.view(batch, time, -1)
        
        # Pass hidden state (hx) into the RNN and return the new state
        r_out, new_hx = self.liquid_rnn(r_in, hx)
        
        return self.output(r_out), new_hx