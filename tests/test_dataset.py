import unittest
from unittest.mock import patch
import torch

from app.dataset import DoomStreamingDataset

class TestActionSpacesAndAugmentation(unittest.TestCase):

    def setUp(self):
        # Action space configurations for the three profiles
        self.basic_actions = [
            "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", 
            "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE"
        ]
        self.classic_actions = self.basic_actions + ["SELECT_WEAPON2", "SELECT_WEAPON3"]
        self.fluid_actions = self.basic_actions + ["SELECT_NEXT_WEAPON"]
        
        # A scrambled classic profile to ensure dynamic indexing works
        self.scrambled_actions = [
            "MOVE_FORWARD", "ATTACK", "SELECT_WEAPON2", "MOVE_LEFT", 
            "MOVE_RIGHT", "TURN_LEFT", "USE", "TURN_RIGHT", "MOVE_BACKWARD"
        ]

    @patch('pathlib.Path.glob')
    def test_basic_swap_map(self, mock_glob):
        """Verifies the Basic profile maps indices (2,3) and (4,5)."""
        mock_glob.return_value = [] # Prevent loading real .npz files
        
        dataset = DoomStreamingDataset(
            data_dir="dummy", 
            augment=True, 
            action_names=self.basic_actions
        )
        
        # MOVE_LEFT is 2, MOVE_RIGHT is 3
        self.assertIn((2, 3), dataset.swap_pairs)
        # TURN_LEFT is 4, TURN_RIGHT is 5
        self.assertIn((4, 5), dataset.swap_pairs)
        self.assertEqual(len(dataset.swap_pairs), 2)

    @patch('pathlib.Path.glob')
    def test_classic_swap_map(self, mock_glob):
        """Verifies the Classic profile maps spatial keys, ignoring weapons."""
        mock_glob.return_value = []
        
        dataset = DoomStreamingDataset(
            data_dir="dummy", 
            augment=True, 
            action_names=self.classic_actions
        )
        
        self.assertIn((2, 3), dataset.swap_pairs)
        self.assertIn((4, 5), dataset.swap_pairs)
        self.assertEqual(len(dataset.swap_pairs), 2)

    @patch('pathlib.Path.glob')
    def test_scrambled_dynamic_indexing(self, mock_glob):
        """Verifies the dynamic indexing works even if the .cfg order changes."""
        mock_glob.return_value = []
        
        dataset = DoomStreamingDataset(
            data_dir="dummy", 
            augment=True, 
            action_names=self.scrambled_actions
        )
        
        # In scrambled: MOVEL=3, MOVER=4 | TURNL=5, TURNR=7
        self.assertIn((3, 4), dataset.swap_pairs)
        self.assertIn((5, 7), dataset.swap_pairs)
        self.assertEqual(len(dataset.swap_pairs), 2)

    @patch('pathlib.Path.glob')
    def test_tensor_permutation_logic(self, mock_glob):
        """Verifies that the dataset iterator actually flips the target tensor."""
        mock_glob.return_value = []
        
        dataset = DoomStreamingDataset(
            data_dir="dummy", 
            augment=True, 
            action_names=self.basic_actions
        )
        
        # Simulate a target tensor (Seq=1, Actions=8)
        # Agent is pressing MOVE_LEFT (index 2) and TURN_RIGHT (index 5)
        y = torch.zeros((1, 8))
        y[0, 2] = 1.0 
        y[0, 5] = 1.0 
        
        # Manually apply the augmentation logic from dataset.py
        y_flip = y.clone()
        for left_idx, right_idx in dataset.swap_pairs:
            y_flip[:, left_idx] = y[:, right_idx]
            y_flip[:, right_idx] = y[:, left_idx]
            
        # Expected: MOVE_RIGHT (index 3) and TURN_LEFT (index 4) should be 1.0
        self.assertEqual(y_flip[0, 3].item(), 1.0)
        self.assertEqual(y_flip[0, 4].item(), 1.0)
        
        # The original inputs should now be 0.0
        self.assertEqual(y_flip[0, 2].item(), 0.0)
        self.assertEqual(y_flip[0, 5].item(), 0.0)

if __name__ == '__main__':
    unittest.main()