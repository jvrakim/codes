"""
Provides a Laplace-compatible model.

This module contains the `LaplaceModel` class, which is a wrapper around the
base `Network` class to make it compatible with the Laplace-torch library.
"""

from src.common import base_model as models
from src.common.data_utils import MultiInput
import torch


class LaplaceModel(models.Network):
    """
    A wrapper around the base `Network` class to make it compatible with the
    Laplace-torch library.

    This class inherits from the base `Network` class and overrides the
    `forward` method to handle the `MultiInput` dataclass.
    """

    def __init__(self, C_in: int = 2, C: int = 36, num_classes: int = 10, H_in: int = 32, W_in: int = 32, p: float = 0.0):
        super(LaplaceModel, self).__init__(C_in, C, num_classes, H_in, W_in, p)

    def forward(self, inp: MultiInput or torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        This method handles both `MultiInput` dataclasses and tuples as input
        for backward compatibility.

        Args:
            inp (MultiInput or torch.Tensor): The input to the network.

        Returns:
            torch.Tensor: The output logits.
        """
        if hasattr(inp, "fc_input"):
            x_img = inp.cnn_input
            x_scalar = inp.fc_input
        else:
            x_img, x_scalar = inp

        stem_out = self.stem(x_img, x_scalar)
        cell_out = self.reduction_cell(stem_out, stem_out)
        pooled = self.global_pooling(cell_out)
        features = pooled.view(pooled.size(0), -1)
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits