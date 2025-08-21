"""
This module contains the tester for the Laplace method.
"""

import os

import torch
from laplace import Laplace

from src.common.base_tester import BaseTester


class LaplaceTester(BaseTester):
    """
    Tester for the Laplace method.
    """

    def __init__(self, device, save_path):
        super().__init__(device, save_path)

    def test(self, model, train_loader, val_loader, test_loader):
        """
        Tests the Laplace model.
        """
        path = os.path.join(self.save_path, "laplace_model.pt")
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        la = Laplace(
            model,
            "classification",
            subset_of_weights = "last_layer",
            hessian_structure = "full",
        )
        la.load_state_dict(checkpoint["laplace_state_dict"])

        preds = []
        targets = []
        ids = []
        for input, target, id in test_loader:
            input = input.to(device=self.device)
            target = target.to(device=self.device)

            ids.append(id)
            preds.append(la.predictive_samples(input, pred_type="glm", n_samples=10000))
            targets.append(target)

        predictions = torch.cat(preds, dim=1)
        targets = torch.cat(targets)

        results = {
            "ids": ids,
            "predictions": predictions,
            "targets": targets,
        }
        self.save_results(results)
        return results