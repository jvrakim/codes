"""
Defines the building blocks for neural network architectures.

This module provides a dictionary `OPS` that maps operation names to their
corresponding PyTorch modules. It also defines several custom PyTorch modules
that are used as operations in the search space, such as separable convolutions,
dilated convolutions, and factorized reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# A dictionary that maps operation names to their corresponding PyTorch modules.
# Each operation is a lambda function that takes the number of input channels `C`,
# the stride `stride`, and a boolean `affine` as input, and returns a PyTorch module.
OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(
        3, stride=stride, padding=1
    ),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(
            C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False
        ),
        nn.Conv2d(
            C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False
        ),
        nn.BatchNorm2d(C, affine=affine),
    ),
}


class ReLUConvBN(nn.Module):
    """
    A helper module that consists of a ReLU activation, a 2D convolution,
    and a batch normalization layer.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        affine (bool, optional): Whether to use affine transformations in the
            batch normalization layer. Defaults to True.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.op(x)


class DilConv(nn.Module):
    """
    A dilated separable convolution module.

    This module consists of a ReLU activation, a dilated depthwise convolution,
    a pointwise convolution, and a batch normalization layer.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation factor for the convolution.
        affine (bool, optional): Whether to use affine transformations in the
            batch normalization layer. Defaults to True.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.op(x)


class SepConv(nn.Module):
    """
    A separable convolution module.

    This module consists of two separable convolutions, each of which is
    composed of a depthwise convolution and a pointwise convolution.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        affine (bool, optional): Whether to use affine transformations in the
            batch normalization layer. Defaults to True.
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.op(x)


class Identity(nn.Module):
    """An identity operation that returns the input tensor."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x


class Zero(nn.Module):
    """
    An operation that returns a tensor of zeros with the same shape as the
    input tensor. It supports striding by sub-sampling the input tensor.

    Args:
        stride (int): The stride to use for sub-sampling.
    """

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor of zeros.
        """
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """
    A factorized reduction operation that reduces the spatial resolution of
    the input tensor by a factor of 2.

    This module consists of two parallel 1x1 convolutions with a stride of 2,
    followed by a concatenation and a batch normalization layer.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        affine (bool, optional): Whether to use affine transformations in the
            batch normalization layer. Defaults to True.
    """

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.conv_2 = nn.Conv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.relu(x)
        out1 = self.conv_1(x)
        pad_w = x.size(3) % 2
        pad_h = x.size(2) % 2
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        out2 = self.conv_2(x_padded[:, :, 1:, 1:])
        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out