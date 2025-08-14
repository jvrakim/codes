"""
Provides utilities for saving and managing ensemble model checkpoints.

This module contains the `EnsembleSaver` and `EnsembleManager` classes, which
extend the base `Saver` and `Manager` classes to provide functionality
specific to saving and managing ensemble models. It also includes a function
for loading ensemble models from saved checkpoints.
"""

import os
import torch
from src.common.base_saver import Saver, Manager
from typing import List


class EnsembleSaver(Saver):
    """
    A class for saving ensemble model checkpoints.

    This class extends the base `Saver` class to handle the saving of ensemble
    model checkpoints, which include the parameters and buffers of all models
    in the ensemble.
    """

    def __init__(
        self, save_dir: str = "./", checkpoint_dir: str = "./", filename_prefix: str = None
    ):
        super(EnsembleSaver, self).__init__(
            save_dir, checkpoint_dir, filename_prefix
        )

    def save_checkpoint(
        self,
        epoch: int,
        params: dict,
        buffers: dict,
        optimizer: torch.optim.Optimizer,
        train_losses: list,
        train_accs: list,
        val_losses: list,
        val_accs: list,
        posfix: str = "",
    ):
        """
        Saves a model checkpoint to disk.

        Args:
            epoch (int): The current epoch number (zero-based).
            params (dict): The parameters of the ensemble models.
            buffers (dict): The buffers of the ensemble models.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
            posfix (str, optional): A postfix to add to the checkpoint
                filename. Defaults to "".
        """
        if self.filename_prefix is None:
            filename = f"checkpoint_epoch_{epoch + 1}_{posfix}" + ".pt"
        else:
            filename = (
                self.filename_prefix
                + f"_checkpoint_epoch_{epoch + 1}_{posfix}"
                + ".pt"
            )

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": epoch + 1,
            "params": params,
            "buffers": buffers,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

        torch.save(checkpoint, filepath)

    def save_best(
        self,
        epoch: int,
        params: dict,
        buffers: dict,
        optimizer: torch.optim.Optimizer,
        train_losses: list,
        train_accs: list,
        val_losses: list,
        val_accs: list,
        model_idx: int,
    ):
        """
        Saves the best checkpoint for a specific model in the ensemble.

        Args:
            epoch (int): The current epoch number (zero-based).
            params (dict): The parameters of the ensemble models.
            buffers (dict): The buffers of the ensemble models.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
            model_idx (int): The index of the model in the ensemble.
        """
        filepath = os.path.join(self.save_dir, f"model_{model_idx}_best.pt")

        model_params = {k: v[model_idx] for k, v in params.items()}
        model_buffers = {
            k: v[model_idx] if v.dim() > 0 else v for k, v in buffers.items()
        }

        model_state_dict = {}
        model_state_dict.update(model_params)
        model_state_dict.update(model_buffers)

        checkpoint = {
            "epoch": epoch + 1,
            "model_idx": model_idx,
            "model_state_dict": model_state_dict,
            "params": model_params,
            "buffers": model_buffers,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

        torch.save(checkpoint, filepath)


class EnsembleManager(Manager):
    """
    A class for managing the training of ensemble models.

    This class extends the base `Manager` class to handle the saving and
    evaluation schedule for ensemble models, and keeps track of the best
    model for each member of the ensemble.

    Args:
        num_epochs (int, optional): The total number of training epochs.
            Defaults to 150.
        eval_freq (int, optional): The frequency (in epochs) at which to
            evaluate the model. Defaults to 1.
        save_freq (int, optional): The frequency (in epochs) at which to
            save model checkpoints. Defaults to 10.
        save_dir (str, optional): The directory to save checkpoints in.
            Defaults to "./".
        filename_prefix (str, optional): A prefix to add to the checkpoint
            filenames. Defaults to None.
        num_models (int, optional): The number of models in the ensemble.
            Defaults to 1.
    """

    def __init__(
        self,
        num_epochs: int = 150,
        eval_freq: int = 1,
        save_freq: int = 10,
        save_dir: str = "./",
        filename_prefix: str = None,
        num_models: int = 1,
    ):
        super(EnsembleManager, self).__init__(
            num_epochs, eval_freq, save_freq, save_dir, filename_prefix
        )
        self.saver = EnsembleSaver(
            save_dir=self.save_dir,
            checkpoint_dir=self.checkpoint_dir,
            filename_prefix=self.filename_prefix,
        )
        self.num_models = num_models
        self.best_val_losses = [float("inf")] * num_models
        self.best_epochs = [-1] * num_models

    def save_best(
        self,
        epoch: int,
        params: dict,
        buffers: dict,
        optimizer: torch.optim.Optimizer,
        train_losses: list,
        train_accs: list,
        val_losses: list,
        val_accs: list,
    ):
        """
        Checks and saves the best checkpoint for each model in the ensemble.

        Args:
            epoch (int): The current epoch number (zero-based).
            params (dict): The parameters of the ensemble models.
            buffers (dict): The buffers of the ensemble models.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
        """
        current_val_losses = val_losses[-1]

        for model_idx in range(self.num_models):
            current_loss = current_val_losses[model_idx].item()

            if current_loss < self.best_val_losses[model_idx]:
                self.best_val_losses[model_idx] = current_loss
                self.best_epochs[model_idx] = epoch
                self.saver.save_best(
                    epoch,
                    params,
                    buffers,
                    optimizer,
                    train_losses,
                    train_accs,
                    val_losses,
                    val_accs,
                    model_idx,
                )

    def save_checkpoint(
        self,
        epoch: int,
        params: dict,
        buffers: dict,
        optimizer: torch.optim.Optimizer,
        train_losses: list,
        train_accs: list,
        val_losses: list,
        val_accs: list,
        posfix: str = "",
    ):
        """
        Saves a checkpoint and deletes the old one.

        Args:
            epoch (int): The current epoch number (zero-based).
            params (dict): The parameters of the ensemble models.
            buffers (dict): The buffers of the ensemble models.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
            posfix (str, optional): A postfix to add to the checkpoint
                filename. Defaults to "".
        """
        self.saver.save_checkpoint(
            epoch,
            params,
            buffers,
            optimizer,
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            posfix=posfix,
        )
        self._delete_old_checkpoint(epoch, posfix=posfix)


def load_ensemble_models(filepath: str, model_list: List[torch.nn.Module]) -> List[torch.nn.Module]:
    """
    Loads the state dicts of the best models in an ensemble from a directory.

    Args:
        filepath (str): The path to the directory containing the saved models.
        model_list (list of torch.nn.Module): A list of the models to load the
            state dicts into.

    Returns:
        list of torch.nn.Module: The list of models with the loaded state
            dicts.
    """
    num_models = len(model_list)

    for model_idx in range(num_models):
        path = os.path.join(filepath, f"model_{model_idx}_best.pt")
        checkpoint = torch.load(path, weights_only=False)
        model_list[model_idx].load_state_dict(checkpoint["model_state_dict"])

    return model_list