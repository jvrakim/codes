"""
Provides metric computation for Laplace models.

This module contains the `LaplaceMetric` class, which extends the
`BaseMetric` to provide metric computation specifically for Laplace models.
"""

import torch
from typing import Dict, Tuple
from src.common.base_metrics import BaseMetric


class LaplaceMetric(BaseMetric):
    """
    A class for computing metrics for Laplace models.

    This class extends the `BaseMetric` to provide metric computation that is
    specific to Laplace models, such as calculating metrics based on the mean
    of the posterior predictive distribution.

    Args:
        device (torch.device): The device to use for computations.
    """

    def __init__(self, device: torch.device):
        super(LaplaceMetric, self).__init__(device=device)

    def get_metrics(
        self, samples: torch.Tensor, target: torch.Tensor, n_bins: int = 15
    ) -> Dict:
        """
        Calculates all the metrics for the Laplace model.

        Args:
            samples (torch.Tensor): The samples from the posterior predictive
                distribution.
            target (torch.Tensor): The target labels.
            n_bins (int, optional): The number of bins to use for the
                calibration metrics. Defaults to 15.

        Returns:
            dict: A dictionary containing all the metrics for each prediction.
        """
        samples = samples.to(device="cpu")
        target = target.to(device="cpu")
        probs = samples.mean(dim=0).to(device="cpu")
        predicted = torch.argmax(probs, dim=-1).to(device="cpu")
        correct_predictions = (predicted == target).float().to(device="cpu")
        confidence = self.get_confidence_scores(probs, predicted)
        (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = self.compute_entropies(probs, samples)

        ECE = self.compute_ECE(confidence, correct_predictions, n_bins=n_bins)
        aECE = self.compute_aECE(
            confidence, correct_predictions, n_bins=n_bins
        )
        AUROC = self.compute_AUROC(confidence, correct_predictions)
        AURC = self.compute_AURC(confidence, correct_predictions)

        results = {
            "predictions": predicted,
            "probabilities": probs,
            "confidence": confidence,
            "correct_predictions": correct_predictions,
            "predictive_entropy": predictive_entropy,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "target": target,
            "ECE": ECE,
            "aECE": aECE,
            "AUROC": AUROC,
            "AURC": AURC,
        }

        return results

    def compute_entropies(
        self, probs: torch.Tensor, samples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the predictive entropy, aleatoric uncertainty, and epistemic
        uncertainty.

        Args:
            probs (torch.Tensor): The mean of the posterior predictive
                distribution.
            samples (torch.Tensor): The samples from the posterior predictive
                distribution.

        Returns:
            tuple: A tuple containing the predictive entropy, aleatoric
                uncertainty, and epistemic uncertainty.
        """
        predictive_entropy = self._predictive_entropy(probs)
        aleatoric_uncertainty = self._aleatoric_uncertainty(samples)
        epistemic_uncertainty = self._epistemic_uncertainty(
            predictive_entropy, aleatoric_uncertainty
        )
        return (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )

    def _aleatoric_uncertainty(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Calculates the aleatoric uncertainty.

        This is the average of the individual model entropies.

        Args:
            samples (torch.Tensor): The samples from the posterior predictive
                distribution.

        Returns:
            torch.Tensor: The aleatoric uncertainty.
        """
        individual_entropies = -torch.sum(
            samples * torch.log(samples + 1e-8), dim=-1
        )
        return torch.mean(individual_entropies, dim=0)

    def compute_credible_intervals(
        self, probs: torch.Tensor, confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the credible intervals for the predicted classes.

        NOTE: This method is under construction.

        Args:
            probs (torch.Tensor): The probabilities from the posterior
                predictive distribution.
            confidence (float, optional): The confidence level for the
                credible intervals. Defaults to 0.95.

        Returns:
            tuple: A tuple containing the lower bound, upper bound, and width
                of the credible intervals.
        """
        pass