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
        # Ensure alpha is a tensor if passed as a list or float
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        
        # We register it as a buffer so it automatically moves to the correct device (CPU/GPU/MPS)
        self.register_buffer('alpha', alpha)
        
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss) 
        
        # Calculate alpha_t
        # If target is 1, alpha_t = alpha. If target is 0, alpha_t = (1 - alpha).
        # This broadcasting works whether self.alpha is a scalar or an 8D vector.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply the mathematically correct focal factor
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Decouples the focusing parameters for positive and negative classes. Heavily penalizes easy negatives (high gamma_neg) while retaining gradients for rare positive actions (low gamma_pos). Includes a margin clip to completely discard the easiest negative predictions.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        probabilities = torch.sigmoid(inputs)
        
        # Calculate base log probabilities safely
        log_probs_pos = torch.log(probabilities.clamp(min=self.eps))
        log_probs_neg = torch.log((1 - probabilities).clamp(min=self.eps))

        # Asymmetric Clipping: Shift the prediction probability for the NEGATIVE loss
        # We shift p (the probability of the positive class) downward by the margin.
        probs_neg_shifted = probabilities.clone()
        if self.clip > 0:
            probs_neg_shifted = (probs_neg_shifted - self.clip).clamp(min=0)

        # Apply asymmetric focal weights
        # Positive loss modulates based on (1-p)
        loss_pos = -targets * (1 - probabilities)**self.gamma_pos * log_probs_pos
        # Negative loss modulates based on shifted p
        loss_neg = -(1 - targets) * (probs_neg_shifted)**self.gamma_neg * log_probs_neg

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss