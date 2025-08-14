"""
Provides utilities for training Laplace models.

This module contains the `LaplaceTrainingCalculator` and `LaplaceTrainer`
classes, which are used to train Laplace models. It also includes a function
for getting model predictions.
"""

import torch
from tqdm import tqdm
from src.common import base_trainer as train_utils
from src.common.data_utils import MultiInput
from typing import Tuple


class LaplaceTrainingCalculator(train_utils.TrainingCalculator):
    """
    A training calculator for Laplace models.

    This class extends the base `TrainingCalculator` to handle the computation
    of loss and accuracy for Laplace models.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        inputs: MultiInput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for a given model and inputs.

        Args:
            model (torch.nn.Module): The model to use.
            criterion (torch.nn.Module): The loss function.
            inputs (MultiInput): The input to the model.
            target (torch.Tensor): The target tensor.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        outputs = model(inputs)
        loss = criterion(outputs, target)
        return loss, outputs

    def compute_accuracy(
        self, outputs: torch.Tensor, target: torch.Tensor, n: int
    ) -> torch.Tensor:
        """
        Computes the accuracy for a given set of outputs and targets.

        Args:
            outputs (torch.Tensor): The model outputs.
            target (torch.Tensor): The target tensor.
            n (int): The number of samples.

        Returns:
            torch.Tensor: The accuracy.
        """
        _, predicted = torch.max(outputs, -1)
        correct = predicted.eq(target).sum(dim=-1)
        accuracy = correct / n
        return accuracy


class LaplaceTrainer(train_utils.Trainer):
    """
    A trainer for Laplace models.

    This class extends the base `Trainer` to handle the training of Laplace
    models.

    Args:
        device (torch.device): The device to use for training.
    """

    def __init__(self, device: torch.device):
        super(LaplaceTrainer, self).__init__(device)
        self.computer = LaplaceTrainingCalculator()

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: MultiInput,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single training step.

        Args:
            model (torch.nn.Module): The model to train.
            inputs (MultiInput): The input to the model.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            grad_clip (float): The value to clip the gradients at.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        optimizer.zero_grad()
        loss, outputs = self.computer.compute_loss(
            model, criterion, inputs, target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]["params"], grad_clip
        )
        optimizer.step()
        return loss, outputs

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Trains the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (torch.nn.Module): The loss function.
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
            for multi_input, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                multi_input = multi_input.to(self.device)
                target = target.to(self.device)
                n = len(multi_input)

                loss, outputs = self.train_step(
                    model, multi_input, target, criterion, optimizer, grad_clip
                )
                accuracy = self.computer.compute_accuracy(outputs, target, n)

                self.loss_tracker.update(loss, n)
                self.acc_tracker.update(accuracy, n)

                tepoch.set_postfix(
                    {
                        "loss": self.loss_tracker.print("{:.3f}"),
                        "accuracy": self.acc_tracker.print("{:.1f}"),
                    }
                )

        return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()

    def evaluate_step(
        self,
        model: torch.nn.Module,
        inputs: MultiInput,
        target: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single evaluation step.

        Args:
            model (torch.nn.Module): The model to evaluate.
            inputs (MultiInput): The input to the model.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        loss, outputs = self.computer.compute_loss(
            model, criterion, inputs, target
        )
        return loss, outputs

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        return_outputs: bool = False,
    ) -> tuple:
        """
        Evaluates the model.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (torch.nn.Module): The loss function.
            epoch (int): The current epoch number.
            return_outputs (bool, optional): Whether to return the model
                outputs. Defaults to False.

        Returns:
            tuple: A tuple containing the average loss and accuracy, and
                optionally the model outputs.
        """
        self.loss_tracker.reset()
        self.acc_tracker.reset()
        model.eval()

        if return_outputs:
            all_outputs = []

        with torch.no_grad():
            with tqdm(dataloader, unit="batch", leave=False) as tepoch:
                for multi_input, target in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    multi_input = multi_input.to(self.device)
                    target = target.to(self.device)
                    n = len(multi_input)

                    loss, outputs = self.evaluate_step(
                        model, multi_input, target, criterion
                    )
                    accuracy = self.computer.compute_accuracy(
                        outputs, target, n
                    )

                    self.loss_tracker.update(loss, n)
                    self.acc_tracker.update(accuracy, n)

                    if return_outputs:
                        all_outputs.append(outputs)

                    tepoch.set_postfix(
                        {
                            "loss": self.loss_tracker.print("{:.3f}"),
                            "accuracy": self.acc_tracker.print("{:.1f}"),
                        }
                    )

        if return_outputs:
            return (
                self.loss_tracker.avg.item(),
                self.acc_tracker.avg.item(),
                torch.cat(all_outputs, dim=0).cpu(),
            )
        else:
            return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()


def get_model_predictions(
    la: torch.nn.Module, test_loader: torch.utils.data.DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets model predictions for a test set.

    Args:
        la (torch.nn.Module): The trained Laplace model.
        test_loader (torch.utils.data.DataLoader): The dataloader for the
            test set.

    Returns:
        tuple: A tuple containing the probabilities and targets.
    """
    all_probs = []
    all_targets = []

    for x, y in test_loader:
        probs = la(x)
        all_probs.append(probs)
        all_targets.append(y)

    return torch.cat(all_probs, dim=1), torch.cat(all_targets)