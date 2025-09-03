"""
This module provides a specialized parser for posterior network experiments.

It extends the BaseExperimentParser with arguments specific to posterior
network methods, allowing for easy configuration of posterior network-related
hyperparameters.
"""

from src.common.base_parser import BaseExperimentParser

class PosteriorParser(BaseExperimentParser):
    """
    A parser for posterior network experiments.

    It inherits from the BaseExperimentParser and adds arguments specific to
    posterior network methods.
    """

    def __init__(self, description="Posterior Network Experiment"):
        super().__init__(description)
        self._add_specific_arguments()

    def _add_specific_arguments(self):
        """
        Adds posterior network-specific arguments to the parser.
        """
        self.parser.add_argument(
            "--regr",
            type=float,
            default=0.01,
            help="The regularization parameter for the posterior network.",
        )
