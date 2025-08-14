"""
Provides custom PyTorch modules for building neural networks.

This module contains various custom PyTorch modules, such as `HexConv2d` and
`HexConv3d`, which are convolutional layers with hexagonal masks, as well as
other common neural network layers like `AvgPool3x3`, `DilConv`, `DilConv5x5`,
and `SkipConnect`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexConv2d(nn.Conv2d):
    """
    A 2D convolutional layer with a hexagonal mask.

    This layer applies a hexagonal mask to the convolutional weights before
    performing the convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolutional kernel.
        mask_type (str, optional): The type of hexagonal mask to use.
            Can be either '3' for a 3x3 mask or '5' for a 5x5 mask.
            Defaults to "3".
        stride (int or tuple, optional): The stride of the convolution.
            Defaults to 1.
        padding (int or tuple, optional): The padding for the convolution.
            Defaults to 0.
        dilation (int or tuple, optional): The dilation for the convolution.
            Defaults to 1.
        groups (int, optional): The number of groups for the convolution.
            Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to
            True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask_type="3",
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(HexConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size

        if mask_type == "3":
            if kH != 3 or kW != 3:
                raise ValueError("For mask_type '3', kernel_size must be 3x3.")
            mask = torch.tensor(
                [[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32
            )
        elif mask_type == "5":
            if kH != 5 or kW != 5:
                raise ValueError("For mask_type '5', kernel_size must be 5x5.")
            mask = torch.tensor(
                [
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError("mask_type must be either '3' or '5'")

        self.register_buffer("mask", mask.view(1, 1, kH, kW))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class HexConv3d(nn.Conv3d):
    """
    A 3D convolutional layer with a hexagonal mask.

    This layer applies a hexagonal mask to the convolutional weights before
    performing the convolution. The 2D mask is repeated along the depth
    dimension to create a 3D mask.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolutional kernel.
        mask_type (str, optional): The type of hexagonal mask to use.
            Can be either '3' for a 3x3 mask or '5' for a 5x5 mask.
            Defaults to "3".
        stride (int or tuple, optional): The stride of the convolution.
            Defaults to 1.
        padding (int or tuple, optional): The padding for the convolution.
            Defaults to 0.
        dilation (int or tuple, optional): The dilation for the convolution.
            Defaults to 1.
        groups (int, optional): The number of groups for the convolution.
            Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to
            True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask_type="3",
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(HexConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size

        if mask_type == "3":
            if kH != 3 or kW != 3:
                raise ValueError(
                    "Kernel height and width must be 3 for mask type '3'"
                )
            mask2d = torch.tensor(
                [[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32
            )
        elif mask_type == "5":
            if kH != 5 or kW != 5:
                raise ValueError(
                    "Kernel height and width must be 5 for mask type '5'"
                )
            mask2d = torch.tensor(
                [
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError("mask_type must be either '3' or '5'")

        full_mask = mask2d.unsqueeze(0).repeat(kD, 1, 1)
        self.register_buffer("mask", full_mask.view(1, 1, kD, kH, kW))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        masked_weight = self.weight * self.mask
        return F.conv3d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class AvgPool3x3(nn.Module):
    """
    A 3x3 average pooling layer followed by batch normalization.

    Args:
        C (int): The number of input and output channels.
        stride (int, optional): The stride of the pooling. Defaults to 1.
    """

    def __init__(self, C: int, stride: int = 1):
        super(AvgPool3x3, self).__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.BatchNorm2d(C, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.op(x)


class DilConv(nn.Module):
    """
    A depthwise separable dilated convolution layer.

    This layer consists of a ReLU activation, a depthwise convolution, a
    pointwise convolution, and a batch normalization layer.

    Args:
        C_in (int): The number of input channels.
        C_out (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding for the convolution.
        dilation (int): The dilation for the convolution.
        affine (bool, optional): Whether to use affine transformations in the
            batch normalization layer. Defaults to True.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.op(x)


class DilConv5x5(nn.Module):
    """
    A 5x5 dilated convolution layer.

    This layer is a wrapper around the `DilConv` class with a kernel size of 5
    and a dilation of 2.

    Args:
        C_in (int): The number of input channels.
        C_out (int): The number of output channels.
        stride (int, optional): The stride of the convolution. Defaults to 1.
    """

    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super(DilConv5x5, self).__init__()
        self.dilconv = DilConv(
            C_in,
            C_out,
            kernel_size=5,
            stride=stride,
            padding=4,
            dilation=2,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.dilconv(x)


class SkipConnect(nn.Module):
    """A skip connection layer."""

    def __init__(self):
        super(SkipConnect, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x