"""
Provides the posterior network model.

This module contains the `PosteriorNetwork` class, which is the main model for
the posterior network method.
"""

import torch
from torch import nn
from src.training_methods.posterior.normalizing_flow_density import (
    NormalizingFlowDensity,
)
from src.common.base_model import Network
from typing import Tuple


class PosteriorNetwork(nn.Module):
    """
    A posterior network model.

    This model uses a feature extractor to extract latent features from the
    input, and then uses a normalizing flow to estimate the density of the
    latent features for each class. The output of the model is the alpha
    parameters of a Dirichlet distribution.

    Args:
        N (torch.Tensor): A tensor containing the count of data from each
            class in the training set.
        C_in (int, optional): The number of input channels for the CNN.
            Defaults to 2.
        C (int, optional): The base channel dimension. Defaults to 36.
        H_in (int, optional): The input height. Defaults to 32.
        W_in (int, optional): The input width. Defaults to 32.
        output_dim (int, optional): The number of classes. Defaults to 10.
        latent_dim (int, optional): The latent dimension. Defaults to 10.
        n_density (int, optional): The number of flow layers. Defaults to 8.
        budget_function (str, optional): The budget function to apply to the
            class counts. Can be 'id', 'id_normalized', 'log', or 'one'.
            Defaults to "id".
        seed (int, optional): The random seed. Defaults to 123.
    """

    def __init__(
        self,
        N: torch.Tensor,
        C_in: int = 2,
        C: int = 36,
        H_in: int = 32,
        W_in: int = 32,
        output_dim: int = 10,
        latent_dim: int = 10,
        n_density: int = 8,
        budget_function: str = "id",
        seed: int = 123,
    ):
        super().__init__()

        torch.cuda.manual_seed(seed)

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.n_density = n_density

        if budget_function == "id":
            self.N = N
        elif budget_function == "id_normalized":
            self.N = N / N.sum()
        elif budget_function == "log":
            self.N = torch.log(N + 1.0)
        elif budget_function == "one":
            self.N = torch.ones_like(N)
        else:
            raise NotImplementedError(
                f"Budget function {budget_function} not supported"
            )

        self.budget_function = budget_function

        self.feature_extractor = Network(
            C_in=C_in,
            C=C,
            num_classes=latent_dim,
            H_in=H_in,
            W_in=W_in,
        )

        self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)

        self.density_estimation = nn.ModuleList(
            [
                NormalizingFlowDensity(
                    dim=self.latent_dim,
                    flow_length=n_density,
                    flow_type="radial_flow",
                )
                for _ in range(self.output_dim)
            ]
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x_img: torch.Tensor, x_scalar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the posterior network.

        Args:
            x_img (torch.Tensor): The image input tensor.
            x_scalar (torch.Tensor): The scalar input tensor.

        Returns:
            tuple: A tuple containing the alpha parameters of the Dirichlet
                distribution and the normalized probabilities.
        """
        batch_size = x_img.size(0)

        if self.N.device != x_img.device:
            self.N = self.N.to(x_img.device)

        N = self.N

        zk = self.feature_extractor(x_img, x_scalar)
        zk = self.batch_norm(zk)

        log_q_zk = torch.zeros((batch_size, self.output_dim), device=zk.device)
        alpha = torch.zeros((batch_size, self.output_dim), device=zk.device)

        for c in range(self.output_dim):
            log_p = self.density_estimation[c].log_prob(zk)
            log_q_zk[:, c] = log_p
            alpha[:, c] = 1.0 + (N[c] * torch.exp(log_q_zk[:, c]))

        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)

        return alpha, soft_output_pred

    def get_model_info(self) -> dict:
        """
        Returns information about the model.

        Returns:
            dict: A dictionary containing information about the model,
                including the total number of parameters, the number of
                trainable parameters, the model size, the latent dimension,
                the number of density layers, and the number of output
                classes.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "latent_dim": self.latent_dim,
            "n_density_layers": self.n_density,
            "output_classes": self.output_dim,
        }