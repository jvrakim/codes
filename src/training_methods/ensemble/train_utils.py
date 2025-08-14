"""
Provides utilities for training ensemble models.

This module contains the `EnsembleAverageMeter`, `EnsembleTrainingCalculator`,
and `EnsembleTrainer` classes, which are used to train ensemble models. It
also includes a function for getting the trainable parameters of a model.
"""

import torch
from torch.func import vmap, functional_call
from src.common.base_trainer import AverageMeter, TrainingCalculator, Trainer
from src.common.print_utils import RichProgress
from typing import List, Tuple, Dict


class EnsembleAverageMeter(AverageMeter):
    """
    An average meter for ensemble models.

    This class extends the base `AverageMeter` to handle the tracking of
    metrics for each model in an ensemble.

    Args:
        device (torch.device): The device to use for the tensors.
        size (int): The number of models in the ensemble.
    """

    def __init__(self, device: torch.device, size: int):
        super().__init__(device, size)
        self.reset()

    def print(self, fmt: str) -> List[str]:
        """
        Formats the current value of the meter for each model in the ensemble.

        Args:
            fmt (str): The format string.

        Returns:
            list of str: A list of formatted strings, one for each model.
        """
        string = [fmt.format(val) for val in self.val.flatten()]
        return string


class EnsembleTrainingCalculator(TrainingCalculator):
    """
    A training calculator for ensemble models.

    This class extends the base `TrainingCalculator` to handle the computation
    of loss and accuracy for ensemble models using `vmap`.
    """

    def __init__(self):
        super().__init__()
        self.vectorized_compute_loss = vmap(
            self._compute_loss, in_dims=(0, 0, None, None, None, None, None)
        )

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        base_model: torch.nn.Module,
        cnn_input: torch.Tensor,
        fc_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the model using `functional_call`.

        Args:
            params (dict): The parameters of the model.
            buffers (dict): The buffers of the model.
            base_model (torch.nn.Module): The base model.
            cnn_input (torch.Tensor): The CNN input tensor.
            fc_input (torch.Tensor): The fully connected input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        return functional_call(
            base_model, (params, buffers), (cnn_input, fc_input)
        )

    def _compute_loss(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        base_model: torch.nn.Module,
        cnn_input: torch.Tensor,
        fc_input: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for a single model in the ensemble.

        Args:
            params (dict): The parameters of the model.
            buffers (dict): The buffers of the model.
            base_model (torch.nn.Module): The base model.
            cnn_input (torch.Tensor): The CNN input tensor.
            fc_input (torch.Tensor): The fully connected input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the loss and the model output.
        """
        y_pred = self.forward(params, buffers, base_model, cnn_input, fc_input)
        loss = criterion(y_pred, target)
        return loss, y_pred

    def compute_loss(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        base_model: torch.nn.Module,
        cnn_input: torch.Tensor,
        fc_input: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for all models in the ensemble.

        Args:
            params (dict): The parameters of the ensemble models.
            buffers (dict): The buffers of the ensemble models.
            base_model (torch.nn.Module): The base model.
            cnn_input (torch.Tensor): The CNN input tensor.
            fc_input (torch.Tensor): The fully connected input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the losses and the model outputs for
                all models in the ensemble.
        """
        losses, y_preds = self.vectorized_compute_loss(
            params, buffers, base_model, cnn_input, fc_input, target, criterion
        )
        return losses, y_preds


class EnsembleTrainer(Trainer):
    """
    A trainer for ensemble models.

    This class extends the base `Trainer` to handle the training of ensemble
    models.

    Args:
        device (torch.device): The device to use for training.
        num_models (int): The number of models in the ensemble.
        params (dict): The parameters of the ensemble models.
        buffers (dict): The buffers of the ensemble models.
    """

    def __init__(
        self,
        device: torch.device,
        num_models: int,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
    ):
        super().__init__(device)
        self.multiplier = torch.tensor(num_models, device=device)
        self.model_name = [f"model {i + 1}" for i in range(num_models)]
        self.params = params
        self.buffers = buffers
        self.computer = EnsembleTrainingCalculator()
        self.loss_tracker = EnsembleAverageMeter(device, num_models)
        self.acc_tracker = EnsembleAverageMeter(device, num_models)

    def train_step(
        self,
        model: torch.nn.Module,
        fc_input: torch.Tensor,
        cnn_input: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single training step for the ensemble.

        Args:
            model (torch.nn.Module): The base model.
            fc_input (torch.Tensor): The fully connected input tensor.
            cnn_input (torch.Tensor): The CNN input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            grad_clip (float): The value to clip the gradients at.

        Returns:
            tuple: A tuple containing the losses and the model outputs for
                all models in the ensemble.
        """
        optimizer.zero_grad()
        losses, outputs = self.computer.compute_loss(
            self.params,
            self.buffers,
            model,
            cnn_input,
            fc_input,
            target,
            criterion,
        )
        losses = losses * self.multiplier
        loss = losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]["params"], grad_clip
        )
        optimizer.step()
        return losses / self.multiplier, outputs

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the ensemble for one epoch.

        Args:
            model (torch.nn.Module): The base model.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            grad_clip (float): The value to clip the gradients at.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the average losses and accuracies for
                the epoch.
        """
        model.train()
        self.loss_tracker.reset()
        self.acc_tracker.reset()

        with RichProgress(len(dataloader), self.model_name) as prog:
            for samples in dataloader:
                prog.set_description(f"Epoch {epoch}")
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
                prog.set_metrics(
                    self.model_name,
                    self.loss_tracker.print("{:.3f}"),
                    self.acc_tracker.print("{:.1f}"),
                )
                prog.update()
                prog.refresh()
        return self.loss_tracker.avg, self.acc_tracker.avg

    def evaluate_step(
        self,
        model: torch.nn.Module,
        fc_input: torch.Tensor,
        cnn_input: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single evaluation step for the ensemble.

        Args:
            model (torch.nn.Module): The base model.
            fc_input (torch.Tensor): The fully connected input tensor.
            cnn_input (torch.Tensor): The CNN input tensor.
            target (torch.Tensor): The target tensor.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the losses and the model outputs for
                all models in the ensemble.
        """
        losses, outputs = self.computer.compute_loss(
            self.params,
            self.buffers,
            model,
            cnn_input,
            fc_input,
            target,
            criterion,
        )
        return losses, outputs

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        return_outputs: bool = False,
    ) -> tuple:
        """
        Evaluates the ensemble.

        Args:
            model (torch.nn.Module): The base model.
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            criterion (torch.nn.Module): The loss function.
            epoch (int): The current epoch number.
            return_outputs (bool, optional): Whether to return the model
                outputs and IDs. Defaults to False.

        Returns:
            tuple: A tuple containing the average losses and accuracies, and
                optionally the model outputs and IDs.
        """
        self.loss_tracker.reset()
        self.acc_tracker.reset()
        model.eval()

        if return_outputs:
            all_outputs = []
            all_ids = []
            all_targets = []

        with torch.no_grad():
            with RichProgress(len(dataloader), self.model_name) as prog:
                for samples in dataloader:
                    prog.set_description(f"Epoch {epoch}")
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
                        all_outputs.append(outputs)
                        all_targets.append(target)
                        if "ID" in samples:
                            all_ids.append(samples["ID"])

                    prog.set_metrics(
                        self.model_name,
                        self.loss_tracker.print("{:.3f}"),
                        self.acc_tracker.print("{:.1f}"),
                    )
                    prog.update()
                    prog.refresh()

        if return_outputs:
            return (
                torch.cat(all_targets).cpu(),
                torch.cat(all_outputs, dim=1).cpu(),
                torch.cat(all_ids).cpu(),
            )
        else:
            return self.loss_tracker.avg, self.acc_tracker.avg


def get_trainable_parameters(
    model: torch.nn.Module, num_models: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Gets the trainable parameters and buffers of a model and repeats them for
    each model in the ensemble.

    Args:
        model (torch.nn.Module): The base model.
        num_models (int): The number of models in the ensemble.

    Returns:
        tuple: A tuple containing the parameters and buffers of the ensemble.
    """
    params = {
        name: p.repeat(num_models, *[1] * p.dim())
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    buffers = {
        name: p.repeat(num_models, *[1] * p.dim())
        for name, p in model.named_buffers()
    }
    return params, buffers