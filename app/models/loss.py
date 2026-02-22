import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    r"""
    Focal Loss for multi-label classification.
    
    Dynamically scales the loss based on prediction confidence, heavily penalizing the model for missing rare classes (like shooting) while suppressing the gradient of easily classified, high-frequency classes (like walking forward).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard BCE with logits (unreduced)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # pt is the probability of the true class. 
        # Using the mathematical identity: pt = exp(-BCE)
        pt = torch.exp(-bce_loss) 
        
        # Apply the focal modulating factor
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss