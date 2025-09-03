"""
This module contains the tester for the posterior network method.
"""

import os

import torch

from src.common.base_tester import BaseTester
from src.training_methods.posterior import train_utils


class PosteriorTester(BaseTester):
    """
    Tester for the posterior network method.
    """

    def __init__(self, device, save_path, num_classes, regr):
        super().__init__(device, save_path)
        self.trainer = train_utils.PosteriorTrainer(device, num_classes, regr)

    def test(self, model, test_loader, criterion):
        """
        Tests the posterior network model.
        """
        path = os.path.join(self.save_path, "model_best.pt")
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        targets, alphas, probs, ids = self.trainer.evaluate(
            model, test_loader, criterion, 0, return_outputs=True
        )

        results = {
            "ids": ids,
            "alphas": alphas,
            "probs": probs,
            "targets": targets,
        }
        self.save_results(results)
        return results