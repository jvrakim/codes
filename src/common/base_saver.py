"""
Provides classes for saving and managing model checkpoints.

This module contains the `Saver` class, which handles the saving of model
checkpoints, and the `Manager` class, which manages the saving and evaluation
schedule during training.
"""

import os
import torch


class Saver:
    """
    A class for saving model checkpoints.

    Args:
        save_dir (str, optional): The directory to save the best model
            checkpoint in. Defaults to "./".
        checkpoint_dir (str, optional): The directory to save the periodic
            model checkpoints in. Defaults to "./".
        filename_prefix (str, optional): A prefix to add to the checkpoint
            filenames. Defaults to None.
    """

    def __init__(
        self, save_dir="./", checkpoint_dir="./", filename_prefix=None
    ):
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.filename_prefix = filename_prefix

    def save_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        posfix="",
    ):
        """
        Saves a model checkpoint to disk.

        Args:
            epoch (int): The current epoch number (zero-based).
            model (torch.nn.Module): The model to save.
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

        torch.save(checkpoint, filepath)

    def save_best(
        self,
        epoch,
        model,
        optimizer,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
    ):
        """
        Saves the best model checkpoint to disk with a fixed filename.

        Args:
            epoch (int): The current epoch number (zero-based).
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
        """
        filepath = os.path.join(self.save_dir, "model_best.pt")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }

        torch.save(checkpoint, filepath)


class Manager:
    """
    A class for managing the training process.

    This class handles the evaluation and saving schedule, and keeps track of
    the best model based on validation loss.

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
    """

    def __init__(
        self,
        num_epochs=150,
        eval_freq=1,
        save_freq=10,
        save_dir="./",
        filename_prefix=None,
    ):
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix

        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        self.best_val_loss = float("inf")
        self.best_epoch = -1

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.saver = Saver(
            save_dir=self.save_dir,
            checkpoint_dir=self.checkpoint_dir,
            filename_prefix=self.filename_prefix,
        )

    def should_evaluate(self, epoch: int) -> bool:
        """
        Determines whether to evaluate the model at the given epoch.

        Args:
            epoch (int): The current epoch number (zero-based).

        Returns:
            bool: True if the model should be evaluated, False otherwise.
        """
        return ((epoch + 1) % self.eval_freq == 0) or (
            (epoch + 1) == self.num_epochs
        )

    def should_save(self, epoch: int) -> bool:
        """
        Determines whether to save a model checkpoint at the given epoch.

        Args:
            epoch (int): The current epoch number (zero-based).

        Returns:
            bool: True if a checkpoint should be saved, False otherwise.
        """
        return ((epoch + 1) % self.save_freq == 0) or (
            (epoch + 1) == self.num_epochs
        )

    def _delete_old_checkpoint(self, epoch: int, posfix: str = ""):
        """
        Deletes the previous checkpoint if it exists.

        Args:
            epoch (int): The current epoch number (zero-based).
            posfix (str, optional): A postfix that was added to the
                checkpoint filename. Defaults to "".
        """
        if epoch > 0:
            prev_epoch = epoch - self.save_freq
            if prev_epoch >= 0:
                if self.filename_prefix is None:
                    old_filename = (
                        f"checkpoint_epoch_{prev_epoch + 1}_{posfix}.pt"
                    )
                else:
                    old_filename = (
                        self.filename_prefix
                        + f"_checkpoint_epoch_{prev_epoch + 1}_{posfix}.pt"
                    )
                old_filepath = os.path.join(self.checkpoint_dir, old_filename)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)

    def save_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        posfix="",
    ):
        """
        Saves a checkpoint and deletes the old one.

        Args:
            epoch (int): The current epoch number (zero-based).
            model (torch.nn.Module): The model to save.
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
            model,
            optimizer,
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            posfix=posfix,
        )
        self._delete_old_checkpoint(epoch, posfix=posfix)

    def save_best(
        self,
        epoch,
        model,
        optimizer,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
    ):
        """
        Checks if the current model is the best and saves it if it is.

        Args:
            epoch (int): The current epoch number (zero-based).
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            train_losses (list): A list of training losses.
            train_accs (list): A list of training accuracies.
            val_losses (list): A list of validation losses.
            val_accs (list): A list of validation accuracies.
        """
        current_val_loss = val_losses[-1]
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            self.saver.save_best(
                epoch,
                model,
                optimizer,
                train_losses,
                train_accs,
                val_losses,
                val_accs,
            )


def check_for_exps_folders(directory: str, base: str) -> str:
    """
    Returns a unique directory name in the specified directory.

    Args:
        directory (str): The path to the folder where the new directory will
            be created.
        base (str): The base name for the new directory (e.g., "run").

    Returns:
        str: A unique directory name in the format "baseXX", where XX is a
            two-digit number.
    """
    counter = 1
    while True:
        candidate = f"{base}_{counter:02d}"
        if not os.path.exists(os.path.join(directory, candidate)):
            return candidate
        counter += 1