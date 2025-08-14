"""
Provides metric computation for posterior network models.

This module contains the `PosteriorMetric` class, which extends the
`BaseMetric` to provide metric computation specifically for posterior network
models.
"""

import torch
from typing import Dict, Tuple
from src.common.base_metrics import BaseMetric


class PosteriorMetric(BaseMetric):
    """
    A class for computing metrics for posterior network models.

    This class extends the `BaseMetric` to provide metric computation that is
    specific to posterior network models, such as calculating metrics based on
    the alpha values of the Dirichlet distribution.

    Args:
        device (torch.device): The device to use for computations.
    """

    def __init__(self, device: torch.device):
        super(PosteriorMetric, self).__init__(device=device)

    def get_metrics(
        self,
        alphas: torch.Tensor,
        ids: torch.Tensor,
        target: torch.Tensor,
        n_bins: int = 15,
    ) -> Dict:
        """
        Calculates all the metrics for the posterior network model.

        Args:
            alphas (torch.Tensor): The alpha values of the Dirichlet
                distribution.
            ids (torch.Tensor): The IDs of the samples.
            target (torch.Tensor): The target labels.
            n_bins (int, optional): The number of bins to use for the
                calibration metrics. Defaults to 15.

        Returns:
            dict: A dictionary containing all the metrics for each prediction.
        """
        alphas = alphas.to(device="cpu")
        ids = ids.to(device="cpu")
        target = target.to(device="cpu")

        alpha_zero = alphas.sum(axis=1)
        probs = alphas / alpha_zero[:, None]
        predicted = torch.argmax(probs, dim=-1)
        correct_predictions = (predicted == target).float()
        confidence = self.get_confidence_scores(probs, predicted)
        (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = self.compute_entropies(probs, alphas, alpha_zero)

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
            "ECE": ECE,
            "aECE": aECE,
            "AUROC": AUROC,
            "AURC": AURC,
            "target": target,
            "ids": ids,
        }

        return results

    def compute_entropies(
        self,
        probs: torch.Tensor,
        alphas: torch.Tensor,
        alpha_zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the predictive entropy, aleatoric uncertainty, and epistemic
        uncertainty.

        Args:
            probs (torch.Tensor): The probabilities from the posterior
                network.
            alphas (torch.Tensor): The alpha values of the Dirichlet
                distribution.
            alpha_zero (torch.Tensor): The sum of the alpha values.

        Returns:
            tuple: A tuple containing the predictive entropy, aleatoric
                uncertainty, and epistemic uncertainty.
        """
        predictive_entropy = self._predictive_entropy(probs)
        aleatoric_uncertainty = self._aleatoric_uncertainty(
            probs, alphas, alpha_zero
        )
        epistemic_uncertainty = self._epistemic_uncertainty(
            predictive_entropy, aleatoric_uncertainty
        )
        return (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )

    def _aleatoric_uncertainty(
        self,
        probs: torch.Tensor,
        alphas: torch.Tensor,
        alpha_zero: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the aleatoric uncertainty.

        Args:
            probs (torch.Tensor): The probabilities from the posterior
                network.
            alphas (torch.Tensor): The alpha values of the Dirichlet
                distribution.
            alpha_zero (torch.Tensor): The sum of the alpha values.

        Returns:
            torch.Tensor: The aleatoric uncertainty.
        """
        return -torch.sum(
            probs
            * (
                torch.digamma(alphas + 1)
                - torch.digamma(alpha_zero + 1)[:, None]
            ),
            dim=-1,
        )

    def compute_credible_intervals(
        self, probs: torch.Tensor, confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the credible intervals for the predicted classes.

        NOTE: This method is under construction.

        Args:
            probs (torch.Tensor): The probabilities from the posterior
                network.
            confidence (float, optional): The confidence level for the
                credible intervals. Defaults to 0.95.

        Returns:
            tuple: A tuple containing the lower bound, upper bound, and width
                of the credible intervals.
        """
        pass