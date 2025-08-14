"""
This script trains a model and then applies the Laplace approximation to it.
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from laplace import Laplace
from src.training_methods.laplace import model as laplace_model_module
from src.training_methods.laplace import collate
from src.training_methods.laplace import train_utils
from src.common import data_utils
from src.common import print_utils
from src.common import base_saver as save_utils

from torch.utils.tensorboard import SummaryWriter
from src.common.base_saver import check_for_exps_folders

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
# common arguments
parser.add_argument(
    "--data_path",
    type=str,
    default="./",
    help="path for the data to be used during search",
)
parser.add_argument(
    "--workers", type=int, default=2, help="number of workers to load dataset"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size of the dataloader"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--epochs", type=int, default=500, help="num of training epochs"
)
parser.add_argument(
    "--classes",
    type=int,
    default=2,
    help="number of classes to be predicted by the network",
)
parser.add_argument(
    "--save", type=str, default="./", help="folder to save the experiments"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="random seed for experiment reproducibility",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=5,
    help="maximum size of gradient before clipping",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=1,
    help="number of epochs between two evaluations",
)
parser.add_argument(
    "--checkpoint_freq",
    type=int,
    default=10,
    help="number of epochs between two checkpoint saves",
)
# specific arguments
parser.add_argument(
    "--hessian_freq",
    type=int,
    default=100,
    help="frequency of Hessian computation",
)
parser.add_argument(
    "--subset_of_weights",
    type=str,
    default="all",
    choices=["all", "subnetwork", "last_layer"],
    help="which weights to include in the Laplace approximation",
)
parser.add_argument(
    "--hessian_structure",
    type=str,
    default="full",
    choices=["full", "kron", "diag"],
    help="structure of the Hessian approximation",
)

args = parser.parse_args()

exp_num = check_for_exps_folders(args.save, "laplace_run")
run_filepath = os.path.join(args.save, exp_num)

os.makedirs(run_filepath, exist_ok=True)
writer = SummaryWriter(log_dir=run_filepath)


def main():
    """
    Trains a model and then applies the Laplace approximation to it.
    """
    if not torch.cuda.is_available():
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_classes=args.classes,
            return_dimensions=True,
            return_tuple=True,
            collate_fn=collate.custom_collate_fn,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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
        save_dir=run_filepath,
    )

    train_accs = torch.zeros(args.epochs)
    train_losses = torch.zeros(args.epochs)

    num_eval_points = (args.epochs + args.eval_freq - 1) // args.eval_freq
    val_accs = torch.zeros(num_eval_points)
    val_losses = torch.zeros(num_eval_points)

    # Training loop
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

    path = os.path.join(run_filepath, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    la = Laplace(
        model,
        "classification",
        subset_of_weights = "last_layer",
        hessian_structure = "full",
    )
    la.fit(train_loader)

    la.optimize_prior_precision(
        method = "gridsearch",
        pred_type = "glm",
        link_approx = "mc",
        val_loader = val_loader,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "laplace_state_dict": la.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        },
        os.path.join(run_filepath, "laplace_model.pt"),
    )

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

    torch.save(samples, os.path.join(run_filepath, "test_results.pt"))

    writer.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(duration))
    print(f"The training took {formatted_time}")
