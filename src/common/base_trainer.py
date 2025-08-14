"""
Provides classes for training neural network models.

This module contains the `AverageMeter`, `TrainingCalculator`, and `Trainer`
classes, which are used to train neural network models. The `AverageMeter`
class is a helper class for tracking the average of a value, the
`TrainingCalculator` class is a helper class for computing the loss and
accuracy, and the `Trainer` class is the main class for training models.
"""

import torch
from tqdm import tqdm


class AverageMeter:
    """
    A helper class for tracking the average of a value.

    Args:
        device (torch.device): The device to use for the tensors.
        size (int, optional): The size of the tensor to track. Defaults to 1.
    """

    def __init__(self, device, size=1):
        self.device = device
        self.size = size
        self.reset()

    def reset(self):
        """Resets the meter."""
        self.avg = torch.zeros(self.size, device=self.device)
        self.sum = torch.zeros(self.size, device=self.device)
        self.cnt = torch.zeros(self.size, device=self.device)

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (torch.Tensor): The new value.
            n (int, optional): The number of samples. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def print(self, fmt: str) -> str:
        """
        Formats the current value of the meter.

        Args:
            fmt (str): The format string.

        Returns:
            str: The formatted string.
        """
        return fmt.format(self.val.item())


class TrainingCalculator:
    """A helper class for computing the loss and accuracy."""

    def compute_loss(self, model, criterion, cnn_input, fc_input, target):
        """
        Computes the loss for a given model and inputs.

        Args:
            model (torch.nn.Module): The model to use.
            criterion (torch.nn.Module): The loss function.
            cnn_input (torch.Tensor): The CNN input tensor.
            fc_input (torch.Tensor): The fully connected input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        outputs = model(cnn_input, fc_input)
        loss = criterion(outputs, target)
        return loss, outputs

    def compute_accuracy(self, outputs, target, n):
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


class Trainer:
    """
    A class for training neural network models.

    Args:
        device (torch.device): The device to use for training.
    """

    def __init__(self, device):
        self.device = device
        self.computer = TrainingCalculator()
        self.loss_tracker = AverageMeter(device)
        self.acc_tracker = AverageMeter(device)

    def train_step(
        self,
        model,
        fc_input,
        cnn_input,
        target,
        criterion,
        optimizer,
        grad_clip,
    ):
        """
        Performs a single training step.

        Args:
            model (torch.nn.Module): The model to train.
            fc_input (torch.Tensor): The fully connected input tensor.
            cnn_input (torch.Tensor): The CNN input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            grad_clip (float): The value to clip the gradients at.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        optimizer.zero_grad()
        loss, outputs = self.computer.compute_loss(
            model, criterion, cnn_input, fc_input, target
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]["params"], grad_clip
        )
        optimizer.step()
        return loss, outputs

    def train_epoch(
        self, model, dataloader, criterion, optimizer, grad_clip, epoch
    ):
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
            for samples in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                fc_input, cnn_input, target = (
                    samples["fc_input"].to(self.device),
                    samples["cnn_input"].to(self.device),
                    samples["target"].to(self.device),
                )
                n = fc_input.size(0)
                loss, outputs = self.train_step(
                    model,
                    fc_input,
                    cnn_input,
                    target,
                    criterion,
                    optimizer,
                    grad_clip,
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

    def evaluate_step(self, model, fc_input, cnn_input, target, criterion):
        """
        Performs a single evaluation step.

        Args:
            model (torch.nn.Module): The model to evaluate.
            fc_input (torch.Tensor): The fully connected input tensor.
            cnn_input (torch.Tensor): The CNN input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the loss and the model outputs.
        """
        loss, outputs = self.computer.compute_loss(
            model, criterion, cnn_input, fc_input, target
        )
        return loss, outputs

    def evaluate(
        self, model, dataloader, criterion, epoch, return_outputs=False
    ):
        """
        Evaluates the model.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (torch.nn.Module): The loss function.
            epoch (int): The current epoch number.
            return_outputs (bool, optional): Whether to return the model
                outputs and IDs. Defaults to False.

        Returns:
            tuple: A tuple containing the average loss and accuracy, and
                optionally the model outputs and IDs.
        """
        self.loss_tracker.reset()
        self.acc_tracker.reset()
        model.eval()

        if return_outputs:
            all_outputs = []
            all_ids = []

        with torch.no_grad():
            with tqdm(dataloader, unit="batch", leave=False) as tepoch:
                for samples in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    fc_input, cnn_input, target = (
                        samples["fc_input"].to(self.device),
                        samples["cnn_input"].to(self.device),
                        samples["target"].to(self.device),
                    )
                    n = fc_input.size(0)
                    loss, outputs = self.evaluate_step(
                        model, fc_input, cnn_input, target, criterion
                    )
                    accuracy = self.computer.compute_accuracy(
                        outputs, target, n
                    )
                    self.loss_tracker.update(loss, n)
                    self.acc_tracker.update(accuracy, n)

                    if return_outputs:
                        all_outputs.append(outputs.cpu())
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
                self.loss_tracker.avg.item(),
                self.acc_tracker.avg.item(),
                torch.cat(all_outputs, dim=0),
                torch.cat(all_ids, dim=0) if all_ids else torch.tensor([]),
            )
        else:
            return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()