"""
Defines the architectural building blocks for neural networks.

This module introduces the `Genotype` named tuple, which is used to represent
the learned architecture of a neural network cell. It also provides a list of
`PRIMITIVES`, which are the possible operations that can be part of a cell.
"""

from collections import namedtuple

# Represents the learned architecture of a cell.
#
# A Genotype is composed of four parts:
# - normal: A list of (operation, node_index) tuples for the normal cell.
# - normal_concat: A list of node indices to concatenate for the normal cell output.
# - reduce: A list of (operation, node_index) tuples for the reduction cell.
# - reduce_concat: A list of node indices to concatenate for the reduction cell output.
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

# A list of all possible operations that can be used in a cell.
PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]