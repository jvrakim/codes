"""
Provides the basic building blocks for the neural network models.

This module contains the `Stem`, `ReductionCell`, and `Network` classes, which
are used to construct the neural network models in this project. The `Stem`
class processes the input data, the `ReductionCell` class is a specific cell
with a fixed architecture, and the `Network` class combines these components
to form the final model.
"""

import torch
import torch.nn as nn
from src.common.modules import HexConv2d, AvgPool3x3, DilConv5x5, SkipConnect


class Stem(nn.Module):
    """
    A stem module for processing multimodal input data.

    This module takes both image and scalar data as input, processes them
    through separate convolutional and fully connected layers, and then
    combines them into a single tensor.

    Args:
        in_channel (int): The number of input channels for the image data.
        out_channel (int): The number of output channels for the image data.
        H_in (int): The height of the input image.
        W_in (int): The width of the input image.
        padding (int, optional): The padding for the convolutional layers.
            Defaults to 1.
        kernel_size (int, optional): The kernel size for the convolutional
            layers. Defaults to 3.
        stride (int, optional): The stride for the convolutional layers.
            Defaults to 1.
        in_features (int, optional): The number of input features for the
            scalar data. Defaults to 1.
        out_features (int, optional): The number of output features for the
            scalar data. Defaults to 1.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        H_in,
        W_in,
        padding=1,
        kernel_size=3,
        stride=1,
        in_features=1,
        out_features=1,
    ):
        super(Stem, self).__init__()

        self.H_out = int(
            (
                torch.floor(torch.tensor((H_in + 2 * padding - kernel_size) / stride)) + 1
            ).item()
        )
        self.W_out = int(
            (
                torch.floor(torch.tensor((W_in + 2 * padding - kernel_size) / stride)) + 1
            ).item()
        )

        if self.H_out % 2 == 0:
            pool_H = self.H_out
        else:
            pool_H = self.H_out - 1

        if self.W_out % 2 == 0:
            pool_W = self.W_out
        else:
            pool_W = self.W_out - 1

        self.out_features = out_features

        self.cnn_head = HexConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )

        self.fc_head = nn.Linear(
            in_features, out_features * self.H_out * self.W_out
        )

        self.combined = HexConv2d(
            out_channel + out_features,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(out_channel)

        self.pool = nn.AdaptiveAvgPool2d((pool_H, pool_W))

    def forward(self, x_img, x_scalar):
        """
        Performs a forward pass through the stem module.

        Args:
            x_img (torch.Tensor): The input image tensor.
            x_scalar (torch.Tensor): The input scalar tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        cnn_out = self.cnn_head(x_img)
        fc_out = self.fc_head(x_scalar.view(-1, 1))
        fc_out = fc_out.view(-1, self.out_features, self.H_out, self.W_out)

        combined = torch.cat([cnn_out, fc_out], dim=1)

        out = self.combined(combined)

        out = self.batch_norm(out)

        out = self.pool(out)

        return out


class ReductionCell(nn.Module):
    """
    A reduction cell with a fixed architecture.

    This cell implements the following genotype:
    reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1),
            ('avg_pool_3x3', 0), ('avg_pool_3x3', 2),
            ('avg_pool_3x3', 2), ('skip_connect', 3),
            ('skip_connect', 2), ('skip_connect', 3)]
    reduce_concat=[2, 3, 4, 5]

    Args:
        C (int): The number of input and output channels.
    """

    def __init__(self, C):
        super(ReductionCell, self).__init__()

        # Preprocess the two input tensors.
        self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
        )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
        )

        # Define the operations for each node in the cell.
        self.node2_op1 = AvgPool3x3(C, stride=2)
        self.node2_op2 = DilConv5x5(C, C, stride=2)
        self.node3_op1 = AvgPool3x3(C, stride=2)
        self.node3_op2 = AvgPool3x3(C, stride=1)
        self.node4_op1 = AvgPool3x3(C, stride=1)
        self.node4_op2 = SkipConnect()
        self.node5_op1 = SkipConnect()
        self.node5_op2 = SkipConnect()

    def forward(self, s0, s1):
        """
        Performs a forward pass through the reduction cell.

        Args:
            s0 (torch.Tensor): The output from cell i-2.
            s1 (torch.Tensor): The output from cell i-1.

        Returns:
            torch.Tensor: The concatenated output from nodes 2, 3, 4, and 5.
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        node2 = self.node2_op1(s0) + self.node2_op2(s1)
        node3 = self.node3_op1(s0) + self.node3_op2(node2)
        node4 = self.node4_op1(node2) + self.node4_op2(node3)
        node5 = self.node5_op1(node2) + self.node5_op2(node3)

        return torch.cat([node2, node3, node4, node5], dim=1)


class Network(nn.Module):
    """
    The base neural network model for this project.

    This network consists of a stem, a single reduction cell, and a
    classification head.

    Args:
        C_in (int, optional): The number of input channels. Defaults to 2.
        C (int, optional): The number of channels in the hidden layers.
            Defaults to 36.
        num_classes (int, optional): The number of output classes.
            Defaults to 10.
        H_in (int, optional): The height of the input image. Defaults to 32.
        W_in (int, optional): The width of the input image. Defaults to 32.
        p (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, C_in=2, C=36, num_classes=10, H_in=32, W_in=32, p=0.0):
        super(Network, self).__init__()

        self.stem = Stem(
            in_channel=C_in,
            out_channel=C,
            H_in=H_in,
            W_in=W_in,
            in_features=1,
            out_features=C // 4,
        )

        self.reduction_cell = ReductionCell(C=C)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(C * 4, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights of the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_scalar):
        """
        Performs a forward pass through the network.

        Args:
            x_img (torch.Tensor): The input image tensor.
            x_scalar (torch.Tensor): The input scalar tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        stem_out = self.stem(x_img, x_scalar)
        cell_out = self.reduction_cell(stem_out, stem_out)
        pooled = self.global_pooling(cell_out)
        features = pooled.view(pooled.size(0), -1)
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits

    def get_model_info(self):
        """
        Returns information about the model architecture.

        Returns:
            dict: A dictionary containing information about the model,
                including the total number of parameters, the number of
                trainable parameters, and the model size in megabytes.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }