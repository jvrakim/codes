"""
Defines the neural network model for architecture search.

This module contains the `Network` class, which is the main model used for
neural architecture search. The `Network` is composed of `Cell` modules, which
in turn are composed of `MixedOp` modules. The `MixedOp` module represents a
weighted sum of all possible operations, and the `Cell` module represents a
small directed acyclic graph (DAG) of operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.base_model import Stem
from src.arch_search.pdarts.genotype import PRIMITIVES
from src.arch_search.pdarts.operations import (
    OPS,
    Identity,
    FactorizedReduce,
    ReLUConvBN,
)


class MixedOp(nn.Module):
    """
    A mixed operation module that computes a weighted sum of all possible
    operations.

    Args:
        C (int): Number of input and output channels.
        stride (int): Stride for the operations.
        operations_switches (list of bool): A list of booleans indicating
            which operations are active.
        p (float): Dropout probability.
    """

    def __init__(self, C, stride, operations_switches, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for operation in range(len(operations_switches)):
            if operations_switches[operation]:
                primitive = PRIMITIVES[operation]
                op = OPS[primitive](C, stride, False)
                if "pool" in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)

    def update_p(self):
        """Updates the dropout probability for the identity operation."""
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def forward(self, x, weights):
        """
        Performs a forward pass through the mixed operation.

        Args:
            x (torch.Tensor): The input tensor.
            weights (torch.Tensor): The weights for each operation.

        Returns:
            torch.Tensor: The output tensor, which is a weighted sum of the
                outputs of all operations.
        """
        total_weighted_sum = 0
        for weight, operation in zip(weights, self.m_ops):
            processed_value = operation(x)
            total_weighted_sum += weight * processed_value

        return total_weighted_sum


class Cell(nn.Module):
    """
    A cell module, which is a small directed acyclic graph (DAG) of
    operations.

    Args:
        nodes (int): Number of nodes in the cell.
        multiplier (int): Multiplier for the number of output channels.
        Channels (list of int): A list containing the number of output channels
            of the current cell, the previous cell, and the cell before the
            previous cell.
        reduction (bool): Whether this is a reduction cell.
        reduction_prev (bool): Whether the previous cell was a reduction cell.
        edges_operations_switches (list of list of bool): A list of lists of
            booleans indicating which operations are active for each edge.
        p (float): Dropout probability.
    """

    def __init__(
        self,
        nodes,
        multiplier,
        Channels,
        reduction,
        reduction_prev,
        edges_operations_switches,
        p,
    ):
        super(Cell, self).__init__()

        self.reduction = reduction
        self.p = p
        self.config_C_i = (
            Channels[0].clone()
            if torch.is_tensor(Channels[0])
            else Channels[0]
        )
        self.config_C_i_minus_1 = (
            Channels[1].clone()
            if torch.is_tensor(Channels[1])
            else Channels[1]
        )
        self.config_C_i_minus_2 = (
            Channels[2].clone()
            if torch.is_tensor(Channels[2])
            else Channels[2]
        )
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(
                self.config_C_i_minus_2, self.config_C_i_minus_2, affine=False
            )
        else:
            self.preprocess0 = ReLUConvBN(
                self.config_C_i_minus_2, self.config_C_i, 1, 1, 0, affine=False
            )
        self.preprocess1 = ReLUConvBN(
            self.config_C_i_minus_1, self.config_C_i, 1, 1, 0, affine=False
        )
        self._nodes = nodes
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        edge_count = 0
        for node in range(self._nodes):
            for edge in range(2 + node):
                stride = 2 if reduction and edge < 2 else 1
                op = MixedOp(
                    self.config_C_i,
                    stride,
                    operations_switches=edges_operations_switches[edge_count],
                    p=self.p,
                )
                self.cell_ops.append(op)
                edge_count = edge_count + 1

    def update_p(self):
        """Updates the dropout probability for all mixed operations in the cell."""
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        """
        Performs a forward pass through the cell.

        Args:
            s0 (torch.Tensor): The output of the cell before the previous cell.
            s1 (torch.Tensor): The output of the previous cell.
            weights (torch.Tensor): The weights for each operation in the cell.

        Returns:
            torch.Tensor: The output of the cell.
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for node in range(self._nodes):
            s = sum(
                self.cell_ops[offset + edge](h, weights[offset + edge])
                for edge, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    """
    The main neural network model for architecture search.

    Args:
        C (int): Initial number of channels.
        num_classes (int): Number of output classes.
        layers (int): Number of cells in the network.
        nodes (int, optional): Number of nodes in each cell. Defaults to 4.
        multiplier (int, optional): Multiplier for the number of output
            channels. Defaults to 4.
        stem_multiplier (int, optional): Multiplier for the number of output
            channels of the stem. Defaults to 3.
        edges_operations_switches_normal (list of list of bool, optional):
            A list of lists of booleans indicating which operations are active
            for each edge in the normal cells. Defaults to [].
        edges_operations_switches_reduce (list of list of bool, optional):
            A list of lists of booleans indicating which operations are active
            for each edge in the reduction cells. Defaults to [].
        p (float, optional): Dropout probability. Defaults to 0.0.
        H_in (int, optional): Height of the input image. Defaults to 19.
        W_in (int, optional): Width of the input image. Defaults to 19.
        padding (int, optional): Padding for the stem convolution. Defaults to 1.
        kernel_size (int, optional): Kernel size for the stem convolution.
            Defaults to 3.
        stride (int, optional): Stride for the stem convolution. Defaults to 1.
        in_features (int, optional): Number of input features for the stem.
            Defaults to 1.
        out_features (int, optional): Number of output features for the stem.
            Defaults to 1.
    """

    def __init__(
        self,
        C,
        num_classes,
        layers,
        nodes=4,
        multiplier=4,
        stem_multiplier=3,
        edges_operations_switches_normal=[],
        edges_operations_switches_reduce=[],
        p=0.0,
        H_in=19,
        W_in=19,
        padding=1,
        kernel_size=3,
        stride=1,
        in_features=1,
        out_features=1,
    ):
        super(Network, self).__init__()

        assert multiplier <= nodes

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._nodes = nodes
        self._multiplier = multiplier
        self.p = p
        self.edges_operations_switches_normal = (
            edges_operations_switches_normal
        )
        self.edges_operations_switches_reduce = (
            edges_operations_switches_reduce
        )
        self.sigmoid = nn.Sigmoid()
        switch_ons = edges_operations_switches_normal.sum(dim=1)
        self.switch_on = switch_ons[0]

        reduction_cell_indices = (
            [layers // 3, 2 * layers // 3] if layers > 3 else [layers // 3]
        )

        C_stem_input_img = C
        C_stem_out = stem_multiplier * C

        self.stem = Stem(
            C_stem_input_img,
            C_stem_out,
            H_in,
            W_in,
            padding,
            kernel_size,
            stride,
            in_features,
            out_features,
        )

        self.cells = nn.ModuleList()

        C_i_minus_2 = C_stem_out
        C_i_minus_1 = C_stem_out
        C_i = C

        reduction_prev = False
        for cell_idx in range(layers):
            is_reduction_cell = cell_idx in reduction_cell_indices

            current_C_i = C_i
            if is_reduction_cell:
                current_C_i *= multiplier  # Channels for ops inside this reduction cell are increased

            channels_for_cell_constructor = [
                current_C_i,
                C_i_minus_1,
                C_i_minus_2,
            ]

            switches_for_cell = (
                self.edges_operations_switches_reduce
                if is_reduction_cell
                else self.edges_operations_switches_normal
            )

            current_cell_module = Cell(
                nodes,
                multiplier,
                channels_for_cell_constructor,
                is_reduction_cell,
                reduction_prev,
                switches_for_cell,
                self.p,
            )
            self.cells.append(current_cell_module)
            reduction_prev = is_reduction_cell
            C_i_minus_2 = C_i_minus_1
            C_i_minus_1 = (
                multiplier * current_C_i
            )  # Output of this cell (concatenation of 'multiplier' nodes)

            if is_reduction_cell:  # If this was a reduction cell, the base C for the next segment of normal cells increases
                C_i = current_C_i

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_i_minus_1, num_classes)

        self._initialize_alphas()

    def forward(self, x_img, x_scalar):
        """
        Performs a forward pass through the network.

        Args:
            x_img (torch.Tensor): The input image tensor.
            x_scalar (torch.Tensor): The input scalar tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        s0 = s1 = self.stem(x_img, x_scalar)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def update_p(self):
        """Updates the dropout probability for all cells in the network."""
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()

    def _initialize_alphas(self):
        """Initializes the architecture parameters (alphas)."""
        num_edges = self._nodes * (self._nodes + 3) / 2
        num_ops = self.switch_on

        self.alphas_normal = nn.Parameter(
            torch.FloatTensor(int(num_edges), num_ops)
        )
        self.alphas_reduce = nn.Parameter(
            torch.FloatTensor(int(num_edges), num_ops)
        )
        torch.nn.init.normal_(self.alphas_normal, std=1e-3)
        torch.nn.init.normal_(self.alphas_reduce, std=1e-3)

        self._arch_parameters = nn.ParameterList(
            [
                self.alphas_normal,
                self.alphas_reduce,
            ]
        )

    def arch_parameters(self):
        """
        Returns the architecture parameters of the network.

        Returns:
            list of torch.Tensor: A list containing the architecture
                parameters for the normal and reduction cells.
        """
        return self._arch_parameters