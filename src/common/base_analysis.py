"""
Provides a base class for analyzing model results.

This module contains the `BaseAnalyzer` class, which provides methods for
calculating various metrics and curves from model predictions, such as
calibration curves, ROC curves, and risk-coverage curves.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


class BaseAnalyzer:
    """
    A base class for analyzing model results.

    This class takes a dictionary of results as input and provides methods for
    calculating various metrics and curves.

    Args:
        results (dict): A dictionary containing the model's predictions and
            other relevant information.
        confidence_key (str): The key for the confidence values in the results
            dictionary.
        correct_pred_key (str): The key for the correct prediction flags in
            the results dictionary.
    """

    def __init__(
        self,
        results,
        confidence_key="confidence",
        correct_pred_key="correct_predictions",
    ):
        self.results = results
        self.confidence_key = confidence_key
        self.correct_pred_key = correct_pred_key

    def get_calibration_curve(self, n_bins=15):
        """
        Calculates the calibration curve data.

        Args:
            n_bins (int, optional): The number of bins to use for the
                calibration curve. Defaults to 15.

        Returns:
            tuple: A tuple containing the accuracies and confidences for each
                bin.
        """
        confidences = self.results[self.confidence_key].cpu().numpy()
        correct_predictions = self.results[self.correct_pred_key].cpu().numpy()

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = []
        confidences_in_bins = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Adjust the upper bound for the last bin to be inclusive of 1.0
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
                accuracies.append(0.0)  # Append 0 if no predictions in bin
                confidences_in_bins.append(
                    0.0
                )  # Append 0 if no predictions in bin

        return accuracies, confidences_in_bins

    def get_roc_curve(self):
        """
        Calculates the ROC curve data.

        Returns:
            tuple: A tuple containing the false positive rates, true positive
                rates, and the area under the ROC curve.
        """
        confidences = self.results[self.confidence_key].cpu().numpy()
        correct_predictions = self.results[self.correct_pred_key].cpu().numpy()
        fpr, tpr, _ = roc_curve(correct_predictions, confidences)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def get_risk_coverage_curve(self):
        """
        Calculates the risk-coverage curve data.

        Returns:
            tuple: A tuple containing the coverages, risks, and the area
                under the risk-coverage curve.
        """
        confidences = self.results[self.confidence_key].cpu().numpy()
        correct_predictions = self.results[self.correct_pred_key].cpu().numpy()

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