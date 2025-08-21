"""
This module provides a class-based approach for parsing command-line arguments
for neural network experiments.

It includes a base parser with common arguments and specialized parsers for
different training methods.
"""

import argparse


class BaseExperimentParser:
    """
    A base class for parsing command-line arguments for neural network
    experiments.

    It defines common arguments that are shared across all experiment types.
    """

    def __init__(self, description: str):
        """
        Initializes the parser with a description.

        Args:
            description (str): A brief description of the experiment.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_arguments()

    def _add_common_arguments(self):
        """
        Adds common arguments to the parser that are shared across all
        experiments.
        """
        self.parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=["train", "test", "full"],
            help="The mode to run the experiment in.",
        )
        self.parser.add_argument(
            "--experiment_path",
            type=str,
            help="The path to the experiment folder.",
        )
        self.parser.add_argument(
            "--path_to_data",
            type=str,
            default="./",
            help="The root directory containing all datasets.",
        )
        self.parser.add_argument(
            "--training_dataset",
            type=str,
            help="The filename of the training data. This file should not " \
            "include the '_train.pt' or '_val.pt' suffix.",
        )
        self.parser.add_argument(
            "--test_dataset",
            type=str,
            help="The filename of the test data. TThis file should not " \
            "include the '_test.pt' suffix.",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="The batch size for the dataloader.",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.025,
            help="The initial learning rate.",
        )
        self.parser.add_argument(
            "--workers",
            type=int,
            default=2,
            help="The number of workers to load the dataset.",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=500,
            help="The number of training epochs.",
        )
        self.parser.add_argument(
            "--classes",
            type=int,
            default=2,
            help="The number of classes to be predicted by the network.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="The random seed for experiment reproducibility.",
        )
        self.parser.add_argument(
            "--grad_clip",
            type=float,
            default=5,
            help="The maximum size of the gradient before clipping.",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=1,
            help="The number of epochs between two evaluations.",
        )
        self.parser.add_argument(
            "--checkpoint_freq",
            type=int,
            default=10,
            help="The number of epochs between two checkpoint saves.",
        )

    def parse_args(self):
        """
        Parses the command-line arguments and returns them.

        Returns:
            argparse.Namespace: An object containing the parsed arguments.
        """
        return self.parser.parse_args()