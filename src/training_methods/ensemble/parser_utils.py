"""
This module provides a specialized parser for ensemble-based experiments.

It extends the BaseExperimentParser with arguments specific to ensemble
methods, allowing for easy configuration of ensemble-related hyperparameters.
"""

from src.common.base_parser import BaseExperimentParser

class EnsembleParser(BaseExperimentParser):
    """
    A parser for ensemble-based experiments.

    It inherits from the BaseExperimentParser and adds arguments specific to
    ensemble methods.
    """

    def __init__(self, description="Ensemble Experiment"):
        super().__init__(description)
        self._add_specific_arguments()

    def _add_specific_arguments(self):
        """
        Adds ensemble-specific arguments to the parser.
        """
        self.parser.add_argument(
            "--num_models",
            type=int,
            default=10,
            help="The number of models in the ensemble.",
        )
