"""
Provides a custom collate function for the Laplace-torch library.

This module contains a custom collate function that is compatible with the
Laplace-torch library and can handle both tuple and dictionary-based data
formats.
"""

import torch
from src.common import data_utils
from typing import List, Tuple, Dict, Any


def custom_collate_fn(
    batch: List[Tuple[Any, Any]] or List[Dict[str, Any]]
) -> Tuple[data_utils.MultiInput, torch.Tensor]:
    """
    An optimized collate function for Laplace-torch compatibility.

    This function can handle both tuple-based and dictionary-based data
    formats.

    Args:
        batch (list): A list of samples, where each sample is either a tuple
            of (inputs, target) or a dictionary.

    Returns:
        tuple: A tuple containing a `MultiInput` object and the target tensor.
    """
    if isinstance(batch[0], tuple):
        inputs, targets = zip(*batch)
        fc_inputs, cnn_inputs = zip(*inputs)

        multi_input = data_utils.MultiInput(
            fc_input=torch.stack(fc_inputs), cnn_input=torch.stack(cnn_inputs)
        )
        return multi_input, torch.stack(targets)
    else:
        fc_batch = torch.stack([sample["fc_input"] for sample in batch])
        cnn_batch = torch.stack([sample["cnn_input"] for sample in batch])
        target_batch = torch.stack([sample["target"] for sample in batch])
        IDs = torch.stack([sample["ID"] for sample in batch])

        multi_input = data_utils.MultiInput(
            fc_input = fc_batch, cnn_input = cnn_batch
        )
        return multi_input, target_batch, IDs