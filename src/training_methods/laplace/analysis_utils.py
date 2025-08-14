"""
Provides analysis tools for Laplace models.

This module contains the `LaplaceAnalyzer` class, which extends the
`BaseAnalyzer` to provide analysis tools specifically for Laplace models.
"""

from src.common.base_analysis import BaseAnalyzer


class LaplaceAnalyzer(BaseAnalyzer):
    """
    A class for analyzing the results of a Laplace model.

    This class extends the `BaseAnalyzer` to provide analysis tools that are
    specific to Laplace models.

    Args:
        results (dict): A dictionary containing the model's predictions and
            other relevant information.
    """

    def __init__(self, results):
        super().__init__(results)