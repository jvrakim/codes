"""
Provides a normalizing flow density estimator.

This module contains the `NormalizingFlowDensity` class, which is a
normalizing flow model that can be used to estimate the density of a
distribution.
"""

import torch
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import (
    affine_autoregressive,
)
from torch import nn
import torch.distributions as tdist
from typing import Tuple


class NormalizingFlowDensity(nn.Module):
    """
    A normalizing flow density estimator.

    This class implements a normalizing flow model that can be used to estimate
    the density of a distribution. It supports radial and inverse autoregressive
    flows.

    Args:
        dim (int): The dimension of the distribution.
        flow_length (int): The number of transformations in the flow.
        flow_type (str, optional): The type of flow to use. Can be either
            'radial_flow' or 'iaf_flow'. Defaults to "planar_flow".
    """

    def __init__(self, dim: int, flow_length: int, flow_type: str = "planar_flow"):
        super(NormalizingFlowDensity, self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        if self.flow_type == "radial_flow":
            self.transforms = nn.Sequential(
                *(Radial(dim) for _ in range(flow_length))
            )
        elif self.flow_type == "iaf_flow":
            self.transforms = nn.Sequential(
                *(
                    affine_autoregressive(dim, hidden_dims=[128, 128])
                    for _ in range(flow_length)
                )
            )
        else:
            raise NotImplementedError

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the normalizing flow.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the transformed tensor and the sum of the
                log-determinants of the Jacobians of the transformations.
        """
        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = (
                sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            )
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-probability of a tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The log-probability of the input tensor.
        """
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians
        return log_prob_x