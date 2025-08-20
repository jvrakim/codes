"""
Provides analysis tools for ensemble models.

This module contains the `EnsembleAnalyzer` class, which extends the
`BaseAnalyzer` to provide analysis tools specifically for ensemble models.
"""

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
        super().__init__(
            results,
            confidence_key="confidence_logit",
            correct_pred_key="correct_predictions_logit",
        )
