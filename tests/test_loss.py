import math
import unittest

import torch
import torch.nn.functional as F

from app.models.loss import FocalLossWithLogits, AsymmetricLoss

class TestFocalLoss(unittest.TestCase):
    def setUp(self):
        # Create a deterministic batch of 4 sequences, 8 actions each
        torch.manual_seed(42)
        self.inputs = torch.randn(4, 8, requires_grad=True)
        self.targets = torch.randint(0, 2, (4, 8)).float()
        
    def test_positive_loss_with_extreme_alpha_vector(self):
        # Emulate an extreme empirical distribution:
        # A rare class gets an alpha of 0.99. A common class gets 0.05.
        alpha_vector = torch.tensor([0.99, 0.05, 0.5, 0.1, 0.8, 0.2, 0.9, 0.01])
        
        criterion = FocalLossWithLogits(alpha=alpha_vector, gamma=2.0)
        loss = criterion(self.inputs, self.targets)
        
        # 1. Loss must be strictly positive. If it dips below 0, gradients invert.
        self.assertGreaterEqual(loss.item(), 0.0, "Loss evaluated to a negative number.")
        
        # 2. Gradients must flow without NaNs
        loss.backward()
        self.assertFalse(torch.isnan(self.inputs.grad).any(), "NaN gradients detected in Focal Loss.")

    def test_gamma_zero_equals_bce(self):
        # When gamma=0 and alpha=0.5, Focal Loss should perfectly mimic 0.5 * BCE
        criterion = FocalLossWithLogits(alpha=0.5, gamma=0.0, reduction='none')
        focal_out = criterion(self.inputs, self.targets)
        
        bce_out = F.binary_cross_entropy_with_logits(self.inputs, self.targets, reduction='none')
        expected_out = 0.5 * bce_out
        
        torch.testing.assert_close(focal_out, expected_out, rtol=1e-4, atol=1e-4)


class TestAsymmetricLoss(unittest.TestCase):
    def setUp(self):
        self.gamma_neg = 4.0
        self.gamma_pos = 1.0
        self.clip = 0.05
        self.criterion = AsymmetricLoss(
            gamma_neg=self.gamma_neg, 
            gamma_pos=self.gamma_pos, 
            clip=self.clip,
            reduction='none'
        )

    def test_hard_vs_easy_negative_scaling(self):
        # Target is 0 for both (Negative class)
        target = torch.tensor([0.0, 0.0])
        
        # Create specific logits that map to extreme probabilities after sigmoid
        # Hard Negative: p=0.9 (Network is confidently wrong)
        logit_hard = torch.tensor([math.log(0.9 / 0.1)]) 
        # Easy Negative: p=0.01 (Network is confidently correct)
        logit_easy = torch.tensor([math.log(0.01 / 0.99)])
        
        inputs = torch.cat([logit_hard, logit_easy])
        
        losses = self.criterion(inputs, target)
        loss_hard, loss_easy = losses[0].item(), losses[1].item()
        
        # 1. Hard negative must be penalized significantly more than an easy negative
        self.assertGreater(loss_hard, loss_easy, "Hard negative loss must be strictly greater than easy negative loss.")
        
        # 2. With a clip of 0.05, an easy negative of p=0.01 should evaluate to EXACTLY 0 loss
        self.assertEqual(loss_easy, 0.0, f"Easy negative with p < clip should have 0.0 loss, got {loss_easy}")

    def test_gradient_flow_and_stability(self):
        inputs = torch.randn(10, 8, requires_grad=True)
        targets = torch.randint(0, 2, (10, 8)).float()
        
        loss = self.criterion(inputs, targets).mean()
        loss.backward()
        
        self.assertFalse(torch.isnan(inputs.grad).any(), "NaN gradients detected in Asymmetric Loss.")

if __name__ == '__main__':
    unittest.main()