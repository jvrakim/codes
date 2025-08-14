"""
Provides a base class for computing metrics.

This module contains the `BaseMetric` class, which is an abstract base class
for computing various metrics, such as confidence scores, entropies,
calibration errors, and AUCs.
"""

import torch
from typing import Dict, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import roc_curve, auc


class BaseMetric(ABC):
    """
    An abstract base class for computing metrics.

    This class provides a common interface for computing various metrics from
    model predictions. Subclasses must implement the `get_metrics`,
    `compute_entropies`, and `compute_credible_intervals` methods.

    Args:
        device (torch.device): The device to use for computations.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def get_metrics(self, *args, **kwargs) -> Dict:
        """
        Calculates all the metrics.

        This method should receive the probabilities for each class and return a
        dictionary containing all the metrics for each prediction.

        Returns:
            dict: A dictionary containing all the metrics for each prediction.
        """
        pass

    @abstractmethod
    def compute_entropies(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the entropies.

        Returns:
            tuple: A tuple containing the entropy of the expected values, the
                expected value of the entropies, and the mutual information.
        """
        pass

    def get_confidence_scores(
        self, probabilities, predictions
    ) -> torch.Tensor:
        """
        Computes the confidence scores for the predicted classes.

        Args:
            probabilities (torch.Tensor): The probabilities for each class.
            predictions (torch.Tensor): The predicted classes.

        Returns:
            torch.Tensor: The confidence scores for the predicted classes.
        """
        return probabilities.gather(-1, predictions.unsqueeze(-1)).squeeze()

    @abstractmethod
    def compute_credible_intervals(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the credible intervals for the predicted classes.

        Returns:
            tuple: A tuple containing the lower bound, upper bound, and width
                of the credible intervals.
        """
        pass

    def _predictive_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy of the model's prediction.

        Args:
            probs (torch.Tensor): The probabilities for each class.

        Returns:
            torch.Tensor: The entropy of the model's prediction.
        """
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    def _epistemic_uncertainty(
        self, predictive_entropy: torch.Tensor, aleatoric_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the epistemic uncertainty (mutual information).

        Epistemic uncertainty is calculated as:
        I(y, θ|x) = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]
                  = Entropy of mean predictions - Mean entropy of individual predictions

        Args:
            predictive_entropy (torch.Tensor): The entropy of the mean
                predictions.
            aleatoric_uncertainty (torch.Tensor): The mean entropy of the
                individual predictions.

        Returns:
            torch.Tensor: The epistemic uncertainty.
        """
        return predictive_entropy - aleatoric_uncertainty

    def compute_ECE(
        self,
        confidence: torch.Tensor,
        correct_predictions: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """
        Calculates the Expected Calibration Error (ECE).

        Args:
            confidence (torch.Tensor): The predicted probabilities (confidence
                scores).
            correct_predictions (torch.Tensor): A binary tensor indicating
                whether the predictions are correct (1) or not (0).
            n_bins (int, optional): The number of bins to use for the
                calibration calculation. Defaults to 15.

        Returns:
            float: The ECE value.
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = correct_predictions[in_bin].float().mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    def compute_aECE(
        self,
        confidence: torch.Tensor,
        correct_predictions: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """
        Calculates the Adaptive Expected Calibration Error (AdaECE).

        This method uses equal-mass binning instead of equal-width binning.

        Args:
            confidence (torch.Tensor): The predicted probabilities (confidence
                scores).
            correct_predictions (torch.Tensor): A binary tensor indicating
                whether the predictions are correct (1) or not (0).
            n_bins (int, optional): The number of bins to use for the
                calibration calculation. Defaults to 15.

        Returns:
            float: The AdaECE value.
        """
        sorted_indices = torch.argsort(confidence)
        sorted_probs = confidence[sorted_indices]
        sorted_correct = correct_predictions[sorted_indices]

        n_samples = len(confidence)
        bin_size = n_samples // n_bins

        ada_ece = 0.0

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

            if start_idx < end_idx:
                bin_probs = sorted_probs[start_idx:end_idx]
                bin_correct = sorted_correct[start_idx:end_idx]

                avg_confidence = bin_probs.mean()
                accuracy = bin_correct.float().mean()

                bin_weight = len(bin_probs) / n_samples
                ada_ece += torch.abs(avg_confidence - accuracy) * bin_weight

        return ada_ece.item()

    def compute_AUROC(
        self, confidence: torch.Tensor, correct_predictions: torch.Tensor
    ) -> float:
        """
        Calculates the Area Under the ROC Curve (AUROC).

        Args:
            confidence (torch.Tensor): The predicted probabilities (confidence
                scores).
            correct_predictions (torch.Tensor): A binary tensor indicating
                whether the predictions are correct (1) or not (0).

        Returns:
            float: The AUROC value.
        """
        fpr, tpr, _ = roc_curve(correct_predictions.cpu().numpy(), confidence.cpu().numpy())
        roc_auc = auc(fpr, tpr)

        return float(roc_auc)

    def compute_AURC(
        self, confidence: torch.Tensor, correct_predictions: torch.Tensor
    ) -> float:
        """
        Calculates the Area Under the Risk-Coverage Curve (AURC).

        Note: For risk-coverage curves, a lower AURC is better.

        Args:
            confidence (torch.Tensor): The predicted probabilities (confidence
                scores).
            correct_predictions (torch.Tensor): A binary tensor indicating
                whether the predictions are correct (1) or not (0).

        Returns:
            float: The AURC value.
        """
        coverage_values, risk_values = self._calculate_risk_coverage_curve(
            confidence, correct_predictions
        )
        rc_auc = auc(coverage_values.cpu().numpy(), risk_values.cpu().numpy())

        return float(rc_auc)

    def _calculate_risk_coverage_curve(
        self,
        confidence: torch.Tensor,
        correct_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the risk-coverage curve.

        Risk is defined as the classification error rate on the 'covered'
        subset, and coverage is the fraction of examples with uncertainty less
        than or equal to a threshold.

        Args:
            confidence (torch.Tensor): The predicted probabilities (confidence
                scores).
            correct_predictions (torch.Tensor): A binary tensor indicating
                whether the predictions are correct (1) or not (0).

        Returns:
            tuple: A tuple containing the coverage levels and the error rates
                at each coverage level.
        """
        uncertainty = 1.0 - confidence
        sorted_idx = torch.argsort(uncertainty, descending=False)

        N = correct_predictions.numel()
        coverages = torch.arange(
            1, N + 1, dtype=torch.float32, device=self.device
        ) / N
        
        sorted_correct = correct_predictions[sorted_idx].float()
        cumulative_correct = torch.cumsum(sorted_correct, dim=0)

        risks = (
            torch.arange(1, N + 1, dtype=torch.float32, device=self.device)
            - cumulative_correct
        ) / torch.arange(1, N + 1, dtype=torch.float32, device=self.device)

        return coverages, risks