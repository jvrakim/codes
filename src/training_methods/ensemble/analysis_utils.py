"""
Provides analysis tools for ensemble models.

This module contains the `EnsembleAnalyzer` class, which extends the
`BaseAnalyzer` to provide analysis tools specifically for ensemble models.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from src.common.base_analysis import BaseAnalyzer


class EnsembleAnalyzer(BaseAnalyzer):
    """
    A class for analyzing the results of an ensemble model.

    This class extends the `BaseAnalyzer` to provide analysis tools that are
    specific to ensemble models, such as calculating calibration curves, ROC
    curves, and risk-coverage curves based on the ensemble's predictions.

    Args:
        results (dict): A dictionary containing the model's predictions and
            other relevant information. It should contain the keys
            'confidence_logit' and 'correct_predictions_logit'.
    """

    def __init__(self, results):
        super().__init__(results)

    def get_calibration_curve(self, n_bins=15):
        """
        Calculates the calibration curve data for the ensemble's logit
        predictions.

        Args:
            n_bins (int, optional): The number of bins to use for the
                calibration curve. Defaults to 15.

        Returns:
            tuple: A tuple containing the accuracies and confidences for each
                bin.
        """
        confidences = self.results["confidence_logit"].cpu().numpy()
        correct_predictions = (
            self.results["correct_predictions_logit"].cpu().numpy()
        )

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = []
        confidences_in_bins = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            if bin_upper == 1.0:
                in_bin = (confidences >= bin_lower) & (
                    confidences <= bin_upper
                )
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            if np.sum(in_bin) > 0:
                accuracies.append(np.mean(correct_predictions[in_bin]))
                confidences_in_bins.append(np.mean(confidences[in_bin]))
            else:
                accuracies.append(0.0)
                confidences_in_bins.append(0.0)

        return accuracies, confidences_in_bins

    def get_roc_curve(self):
        """
        Calculates the ROC curve data for the ensemble's logit predictions.

        Returns:
            tuple: A tuple containing the false positive rates, true positive
                rates, and the area under the ROC curve.
        """
        confidences = self.results["confidence_logit"].cpu().numpy()
        correct_predictions = (
            self.results["correct_predictions_logit"].cpu().numpy()
        )
        fpr, tpr, _ = roc_curve(correct_predictions, confidences)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def get_risk_coverage_curve(self):
        """
        Calculates the risk-coverage curve data for the ensemble's logit
        predictions.

        Returns:
            tuple: A tuple containing the coverages, risks, and the area
                under the risk-coverage curve.
        """
        confidences = self.results["confidence_logit"].cpu().numpy()
        correct_predictions = (
            self.results["correct_predictions_logit"].cpu().numpy()
        )

        uncertainty = 1.0 - confidences
        sorted_idx = np.argsort(uncertainty)

        N = len(correct_predictions)
        coverages = np.arange(1, N + 1) / N

        sorted_correct = correct_predictions[sorted_idx]
        cumulative_correct = np.cumsum(sorted_correct)

        risks = (np.arange(1, N + 1) - cumulative_correct) / np.arange(
            1, N + 1
        )
        rc_auc = auc(coverages, risks)

        return coverages, risks, rc_auc