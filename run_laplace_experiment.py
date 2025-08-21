"""
This script provides a unified interface for running Laplace-based
neural network experiments.

It supports training, testing, and running a full experiment (both training and
testing) with a single command.
"""

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from laplace import Laplace

from src.training_methods.laplace.parser_utils import LaplaceParser
from src.training_methods.laplace import model as laplace_model_module
from src.training_methods.laplace import collate
from src.training_methods.laplace import train_utils
from src.common import data_utils
from src.common import print_utils
from src.common import base_saver as save_utils


def train(args):
    """
    Trains a model and applies the Laplace approximation.

    Args:
        args: An argparse.Namespace object containing the experiment
              arguments.
    """
    if not torch.cuda.is_available():
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"laplace_{args.training_dataset}_{timestamp}"
    experiment_path = os.path.join(args.experiment_path, experiment_name)

    training_files_path = os.path.join(experiment_path, "training_files")
    checkpoints_path = os.path.join(training_files_path, "checkpoints")
    best_model_path = os.path.join(training_files_path, "best_model")
    logs_path = os.path.join(training_files_path, "logs")

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_path)

    train_loader, val_loader, _, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path=args.path_to_data,
            training_dataset=args.training_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_classes=args.classes,
            return_dimensions=True,
            return_tuple=True,
            collate_fn=collate.custom_collate_fn,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    model = laplace_model_module.LaplaceModel(
        C_in=image_C,
        C=6,
        H_in=image_H,
        W_in=image_W,
        num_classes=args.classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = train_utils.LaplaceTrainer(device=device)
    manager = save_utils.Manager(
        num_epochs=args.epochs,
        save_freq=args.checkpoint_freq,
        save_dir=training_files_path,
    )

    train_accs = torch.zeros(args.epochs)
    train_losses = torch.zeros(args.epochs)
    num_eval_points = (args.epochs + args.eval_freq - 1) // args.eval_freq
    val_accs = torch.zeros(num_eval_points)
    val_losses = torch.zeros(num_eval_points)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = trainer.train_epoch(
            model, train_loader, criterion, optimizer, args.grad_clip, epoch
        )

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)

        if manager.should_evaluate(epoch):
            val_loss, val_acc = trainer.evaluate(
                model, val_loader, criterion, epoch
            )

            eval_idx = epoch // args.eval_freq
            val_losses[eval_idx] = val_loss
            val_accs[eval_idx] = val_acc

            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            manager.save_best(
                epoch,
                model,
                optimizer,
                train_losses[: epoch + 1],
                train_accs[: epoch + 1],
                val_losses[: eval_idx + 1],
                val_accs[: eval_idx + 1],
            )

        epoch_duration = time.time() - epoch_start
        writer.add_scalar("epoch_duration", epoch_duration, epoch)

        loss_fmt = "{:.3f}"
        acc_fmt = "{:.1f}"

        train_loss_str = loss_fmt.format(train_loss)
        train_acc_str = acc_fmt.format(train_acc * 100)
        val_loss_str = loss_fmt.format(val_loss)
        val_acc_str = acc_fmt.format(val_acc * 100)

        msg = (
            "Training: \n"
            f"Loss = {train_loss_str}\n"
            f"Accuracy = {train_acc_str}\n"
            "\n"
            "Validation: \n"
            f"Loss = {val_loss_str}\n"
            f"Accuracy = {val_acc_str}"
        )

        print_utils.print_msg_box(msg=msg, title=f"Epoch {epoch}")

        if manager.should_save(epoch):
            eval_idx = (epoch) // args.eval_freq
            manager.save_checkpoint(
                epoch,
                model,
                optimizer,
                train_losses[: epoch + 1],
                train_accs[: epoch + 1],
                val_losses[: eval_idx + 1],
                val_accs[: eval_idx + 1],
            )

    path = os.path.join(best_model_path, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    la = Laplace(
        model,
        "classification",
        subset_of_weights=args.subset_of_weights,
        hessian_structure=args.hessian_structure,
    )
    la.fit(train_loader)

    la.optimize_prior_precision(
        method="gridsearch",
        pred_type="glm",
        link_approx="mc",
        val_loader=val_loader,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "laplace_state_dict": la.state_dict(),
        },
        os.path.join(best_model_path, "laplace_model.pt"),
    )

    writer.close()
    print(f"Finished training. Best model saved in {best_model_path}")


def test(args):
    """
    Tests a Laplace-approximated model.

    Args:
        args: An argparse.Namespace object containing the experiment
              arguments.
    """
    if not torch.cuda.is_available():
        sys.exit(1)

    best_model_path = os.path.join(args.experiment_path, "training_files", "best_model")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    test_folder_name = f"{args.test_dataset}_{timestamp}"
    test_path = os.path.join(args.experiment_path, test_folder_name)
    os.makedirs(test_path, exist_ok=True)

    _, _, test_loader, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path=args.path_to_data,
            test_dataset=args.test_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_classes=args.classes,
            return_dimensions=True,
            return_tuple=True,
            collate_fn=collate.custom_collate_fn,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = laplace_model_module.LaplaceModel(
        C_in=image_C,
        C=6,
        H_in=image_H,
        W_in=image_W,
        num_classes=args.classes,
    ).to(device)

    path = os.path.join(best_model_path, "laplace_model.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    la = Laplace(
        model,
        "classification",
        subset_of_weights=args.subset_of_weights,
        hessian_structure=args.hessian_structure,
    )
    la.load_state_dict(checkpoint["laplace_state_dict"])

    preds = []
    targets = []
    ids = []
    for input, target, id in test_loader:
        input = input.to(device=device)
        target = target.to(device=device)

        ids.append(id)
        preds.append(
            la.predictive_samples(input, pred_type="glm", n_samples=10000)
        )
        targets.append(target)

    predictions = torch.cat(preds, dim=1)
    targets = torch.cat(targets)

    samples = {
        "ids": ids,
        "predictions": predictions,
        "targets": targets,
    }

    torch.save(samples, os.path.join(test_path, "test_results.pt"))
    print(f"Finished testing. Test results saved in {test_path}")


def main():
    """
    The main function of the script.

    It parses the command-line arguments and calls the appropriate
    function based on the specified mode.
    """
    parser = LaplaceParser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "full":
        train(args)
        test(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(duration))
    print(f"The script took {formatted_time} to run.")
