"""
Provides analysis tools for posterior network models.

This module contains the `PosteriorAnalyzer` class, which extends the
`BaseAnalyzer` to provide analysis tools specifically for posterior network
models.
"""

from src.common.base_analysis import BaseAnalyzer


class PosteriorAnalyzer(BaseAnalyzer):
    """
    A class for analyzing the results of a posterior network model.

    This class extends the `BaseAnalyzer` to provide analysis tools that are
    specific to posterior network models.

    Args:
        results (dict): A dictionary containing the model's predictions and
            other relevant information.
    """

    def __init__(self, results):
        super().__init__(
            results,
            confidence_key="confidence",
            correct_pred_key="correct_predictions",
        )