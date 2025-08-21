"""
This module contains the tester for the ensemble method.
"""

import torch.nn.functional as F
from torch.func import stack_module_state

from src.common.base_tester import BaseTester
from src.training_methods.ensemble import save_utils, train_utils


class EnsembleTester(BaseTester):
    """
    Tester for the ensemble method.
    """

    def __init__(self, device, save_path, num_models):
        super().__init__(device, save_path)
        self.num_models = num_models

    def test(self, model_list, test_loader, criterion, base_model):
        """
        Tests the ensemble model.
        """
        model_list = save_utils.load_ensemble_models(self.save_path, model_list)
        params, buffers = stack_module_state(model_list)

        predictor = train_utils.EnsembleTrainer(
            self.device, self.num_models, params, buffers
        )
        targets, outputs, ids = predictor.evaluate(
            base_model, test_loader, criterion, 0, return_outputs=True
        )
        probs = F.softmax(outputs.mean(dim=0), dim=1)
        individual_probs = F.softmax(outputs, dim=2)

        results = {
            "ids": ids,
            "probs": probs,
            "targets": targets,
            "individual_probs": individual_probs,
        }
        self.save_results(results)
        return results