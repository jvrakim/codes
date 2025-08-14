"""
Provides metric computation for ensemble models.

This module contains the `EnsembleMetric` class, which extends the
`BaseMetric` to provide metric computation specifically for ensemble models.
"""

import torch
from typing import Dict, Tuple
import torch.nn.functional as F
from src.common.base_metrics import BaseMetric


class EnsembleMetric(BaseMetric):
    """
    A class for computing metrics for ensemble models.

    This class extends the `BaseMetric` to provide metric computation that is
    specific to ensemble models, such as calculating metrics based on both the
    ensemble's logit predictions and the mean of the individual models'
    probability predictions.

    Args:
        device (torch.device): The device to use for computations.
    """

    def __init__(self, device: torch.device):
        super(EnsembleMetric, self).__init__(device=device)

    def get_metrics(
        self,
        probs: torch.Tensor,
        individual_probs: torch.Tensor,
        ids: torch.Tensor,
        target: torch.Tensor,
        n_bins: int = 15,
    ) -> Dict:
        """
        Calculates all the metrics for the ensemble model.

        Args:
            probs (torch.Tensor): The probabilities from the ensemble's logit
                predictions.
            individual_probs (torch.Tensor): The probabilities from the
                individual models in the ensemble.
            ids (torch.Tensor): The IDs of the samples.
            target (torch.Tensor): The target labels.
            n_bins (int, optional): The number of bins to use for the
                calibration metrics. Defaults to 15.

        Returns:
            dict: A dictionary containing all the metrics for each prediction.
        """
        probs = probs.to(device="cpu")
        individual_probs = individual_probs.to(device="cpu")
        ids = ids.to(device="cpu")
        target = target.to(device="cpu")

        # Metrics for logit-based predictions
        predicted_logit = torch.argmax(probs, dim=-1)
        correct_predictions_logit = (predicted_logit == target).float()
        confidence_logit = self.get_confidence_scores(probs, predicted_logit)
        predictive_entropy_logit = self._predictive_entropy(probs)
        ECE_logit = self.compute_ECE(
            confidence_logit, correct_predictions_logit, n_bins=n_bins
        )
        aECE_logit = self.compute_aECE(
            confidence_logit, correct_predictions_logit, n_bins=n_bins
        )
        AUROC_logit = self.compute_AUROC(
            confidence_logit, correct_predictions_logit
        )
        AURC_logit = self.compute_AURC(
            confidence_logit, correct_predictions_logit
        )

        # Metrics for probability-based predictions
        mean_probs = torch.mean(individual_probs, dim=0)
        predicted_prob = torch.argmax(mean_probs, dim=-1)
        correct_predictions_prob = (predicted_prob == target).float()
        confidence_prob = self.get_confidence_scores(
            mean_probs, predicted_prob
        )
        (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ) = self.compute_entropies(individual_probs, mean_probs)
        ECE_prob = self.compute_ECE(
            confidence_prob, correct_predictions_prob, n_bins=n_bins
        )
        aECE_prob = self.compute_aECE(
            confidence_prob, correct_predictions_prob, n_bins=n_bins
        )
        AUROC_prob = self.compute_AUROC(
            confidence_prob, correct_predictions_prob
        )
        AURC_prob = self.compute_AURC(
            confidence_prob, correct_predictions_prob
        )

        results = {
            "predictions_logit": predicted_logit,
            "probabilities_logit": probs,
            "confidence_logit": confidence_logit,
            "correct_predictions_logit": correct_predictions_logit,
            "predictive_entropy_logit": predictive_entropy_logit,
            "ECE_logit": ECE_logit,
            "aECE_logit": aECE_logit,
            "AUROC_logit": AUROC_logit,
            "AURC_logit": AURC_logit,
            "predictions_prob": predicted_prob,
            "probabilities_prob": individual_probs,
            "confidence_prob": confidence_prob,
            "correct_predictions_prob": correct_predictions_prob,
            "predictive_entropy": predictive_entropy,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "ECE_prob": ECE_prob,
            "aECE_prob": aECE_prob,
            "AUROC_prob": AUROC_prob,
            "AURC_prob": AURC_prob,
            "target": target,
            "ids": ids,
        }

        return results

    def compute_entropies(
        self, individual_probs: torch.Tensor, mean_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the predictive entropy, aleatoric uncertainty, and epistemic
        uncertainty.

        Args:
            individual_probs (torch.Tensor): The probabilities from the
                individual models in the ensemble.
            mean_probs (torch.Tensor): The mean of the individual models'
                probabilities.

        Returns:
            tuple: A tuple containing the predictive entropy, aleatoric
                uncertainty, and epistemic uncertainty.
        """
        predictive_entropy = self._predictive_entropy(mean_probs)
        aleatoric_uncertainty = self._aleatoric_uncertainty(individual_probs)
        epistemic_uncertainty = self._epistemic_uncertainty(
            predictive_entropy, aleatoric_uncertainty
        )
        return (
            predictive_entropy,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        )

    def _aleatoric_uncertainty(self, individual_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the aleatoric uncertainty.

        This is the average of the individual model entropies.

        Args:
            individual_probs (torch.Tensor): The probabilities from the
                individual models in the ensemble.

        Returns:
            torch.Tensor: The aleatoric uncertainty.
        """
        individual_entropies = -torch.sum(
            individual_probs * torch.log(individual_probs + 1e-8), dim=-1
        )
        return torch.mean(individual_entropies, dim=0)

    def compute_credible_intervals(
        self, probs: torch.Tensor, confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the credible intervals for the predicted classes.

        NOTE: This method is under construction.

        Args:
            probs (torch.Tensor): The probabilities from the ensemble's logit
                predictions.
            confidence (float, optional): The confidence level for the
                credible intervals. Defaults to 0.95.

        Returns:
            tuple: A tuple containing the lower bound, upper bound, and width
                of the credible intervals.
        """
        ensemble_mean = torch.mean(probs, dim=0)
        ensemble_std = torch.std(probs, dim=0)

        z_score = torch.distributions.Normal(0, 1).icdf(
            torch.tensor((1 + confidence) / 2)
        )

        logit_lower = ensemble_mean - z_score * ensemble_std
        logit_upper = ensemble_mean + z_score * ensemble_std

        prob_lower = F.softmax(logit_lower, dim=-1)
        prob_upper = F.softmax(logit_upper, dim=-1)

        return (prob_lower, prob_upper, prob_upper - prob_lower)