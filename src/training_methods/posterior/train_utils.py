"""
Provides utilities for training posterior network models.

This module contains the `uce_loss` function, the `PosteriorTrainingCalculator`
and `PosteriorTrainer` classes, which are used to train posterior network
models. It also includes a function for saving the alpha values of the
Dirichlet distribution.
"""

import torch
from tqdm import tqdm
from torch.distributions.dirichlet import Dirichlet
from src.common.base_trainer import Trainer, TrainingCalculator
from typing import Tuple


def uce_loss(
    alpha: torch.Tensor,
    soft_output: torch.Tensor,
    output_dim: int,
    regr: float,
) -> torch.Tensor:
    """
    Calculates the UCE loss.

    Args:
        alpha (torch.Tensor): The alpha values of the Dirichlet distribution.
        soft_output (torch.Tensor): The normalized probabilities.
        output_dim (int): The number of classes.
        regr (float): The regularization parameter.

    Returns:
        torch.Tensor: The UCE loss.
    """
    alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, output_dim)
    entropy_reg = Dirichlet(alpha).entropy()
    UCE_loss = torch.sum(
        soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
    ) - regr * torch.sum(entropy_reg)

    return UCE_loss


class PosteriorTrainingCalculator(TrainingCalculator):
    """
    A training calculator for posterior network models.

    This class extends the base `TrainingCalculator` to handle the computation
    of loss and accuracy for posterior network models.

    Args:
        output_dim (int): The number of classes.
        regr (float): The regularization parameter.
    """

    def __init__(self, output_dim: int, regr: float):
        super().__init__()
        self.output_dim = output_dim
        self.regr = regr

    def compute_loss(
        self,
        model: torch.nn.Module,
        criterion: callable,
        cnn_input: torch.Tensor,
        fc_input: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the loss for a given model and inputs.

        Args:
            model (torch.nn.Module): The model to use.
            criterion (callable): The loss function.
            cnn_input (torch.Tensor): The CNN input tensor.
            fc_input (torch.Tensor): The fully connected input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            tuple: A tuple containing the loss and a tuple of the model
                outputs and alpha values.
        """
        alpha, outputs = model(cnn_input, fc_input)
        loss = criterion(alpha, target, self.output_dim, self.regr)
        return loss, (outputs, alpha)


class PosteriorTrainer(Trainer):
    """
    A trainer for posterior network models.

    This class extends the base `Trainer` to handle the training of posterior
    network models.

    Args:
        device (torch.device): The device to use for training.
        output_dim (int): The number of classes.
        regr (float): The regularization parameter.
    """

    def __init__(self, device: torch.device, output_dim: int, regr: float):
        super().__init__(device)
        self.computer = PosteriorTrainingCalculator(output_dim, regr)

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: callable,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Trains the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (callable): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            grad_clip (float): The value to clip the gradients at.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the average loss and accuracy for the
                epoch.
        """
        model.train()
        self.loss_tracker.reset()
        self.acc_tracker.reset()

        with tqdm(dataloader, unit="batch", leave=False) as tepoch:
            for samples in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                fc_input, cnn_input, target, one_hot = (
                    samples["fc_input"].to(self.device),
                    samples["cnn_input"].to(self.device),
                    samples["target"].to(self.device),
                    samples["one_hot"].to(self.device),
                )
                n = fc_input.size(0)
                loss, outputs = self.train_step(
                    model,
                    fc_input,
                    cnn_input,
                    one_hot,
                    criterion,
                    optimizer,
                    grad_clip,
                )
                accuracy = self.computer.compute_accuracy(
                    outputs[0], target, n
                )
                self.loss_tracker.update(loss, n)
                self.acc_tracker.update(accuracy, n)
                tepoch.set_postfix(
                    {
                        "loss": self.loss_tracker.print("{:.3f}"),
                        "accuracy": self.acc_tracker.print("{:.1f}"),
                    }
                )
        return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: callable,
        epoch: int,
        return_outputs: bool = False,
    ) -> tuple:
        """
        Evaluates the model.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (callable): The loss function.
            epoch (int): The current epoch number.
            return_outputs (bool, optional): Whether to return the model
                outputs, alpha values, and IDs. Defaults to False.

        Returns:
            tuple: A tuple containing the average loss and accuracy, and
                optionally the targets, alpha values, model outputs, and IDs.
        """
        self.loss_tracker.reset()
        self.acc_tracker.reset()
        model.eval()

        if return_outputs:
            all_outputs = []
            all_ids = []
            all_alphas = []
            all_targets = []

        with torch.no_grad():
            with tqdm(dataloader, unit="batch", leave=False) as tepoch:
                for samples in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    fc_input, cnn_input, target, one_hot = (
                        samples["fc_input"].to(self.device),
                        samples["cnn_input"].to(self.device),
                        samples["target"].to(self.device),
                        samples["one_hot"].to(self.device),
                    )
                    n = fc_input.size(0)
                    loss, outputs = self.evaluate_step(
                        model, fc_input, cnn_input, one_hot, criterion
                    )
                    accuracy = self.computer.compute_accuracy(
                        outputs[0], target, n
                    )
                    self.loss_tracker.update(loss, n)
                    self.acc_tracker.update(accuracy, n)

                    if return_outputs:
                        all_outputs.append(outputs[0])
                        all_alphas.append(outputs[1])
                        all_targets.append(target)
                        if "ID" in samples:
                            all_ids.append(samples["ID"])

                    tepoch.set_postfix(
                        {
                            "loss": self.loss_tracker.print("{:.3f}"),
                            "accuracy": self.acc_tracker.print("{:.1f}"),
                        }
                    )

        if return_outputs:
            return (
                torch.cat(all_targets).to("cpu"),
                torch.cat(all_alphas, dim=0).to("cpu"),
                torch.cat(all_outputs, dim=0).to("cpu"),
                torch.cat(all_ids).to("cpu", dtype=torch.int32),
            )
        else:
            return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()


def save_alphas(alphas: torch.Tensor, save_dir: str):
    """
    Saves the alpha values of the Dirichlet distribution to a file.

    Args:
        alphas (torch.Tensor): The alpha values to save.
        save_dir (str): The directory to save the file in.
    """
    torch.save(alphas, f"{save_dir}/alphas.pt")