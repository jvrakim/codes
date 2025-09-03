"""
This module provides a specialized parser for Laplace-based experiments.

It extends the BaseExperimentParser with arguments specific to the Laplace
approximation method, allowing for easy configuration of Laplace-related
hyperparameters.
"""

from src.common.base_parser import BaseExperimentParser

class LaplaceParser(BaseExperimentParser):
    """
    A parser for Laplace-based experiments.

    It inherits from the BaseExperimentParser and adds arguments specific to
    the Laplace approximation method.
    """

    def __init__(self, description="Laplace Experiment"):
        super().__init__(description)
        self._add_specific_arguments()

    def _add_specific_arguments(self):
        """
        Adds Laplace-specific arguments to the parser.
        """
        self.parser.add_argument(
            "--subset_of_weights",
            type=str,
            default="last_layer",
            choices=["all", "subnetwork", "last_layer"],
            help="Which weights to include in the Laplace approximation.",
        )
        self.parser.add_argument(
            "--hessian_structure",
            type=str,
            default="full",
            choices=["full", "kron", "diag"],
            help="The structure of the Hessian approximation.",
        )
