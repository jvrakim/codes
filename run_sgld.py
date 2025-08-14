"""
This script trains a model using the SGLD optimizer.
"""

import os
import sys
import time
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.training_methods.sgld import optim
from src.training_methods.sgld import save_utils
from src.common import base_model as models
from src.common import data_utils
from src.common import base_trainer as train_utils
from src.common import print_utils

from torch.utils.tensorboard import SummaryWriter
from src.common.base_saver import check_for_exps_folders

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--data_path",
    type=str,
    default="./",
    help="path for the data to be used duing search",
)
parser.add_argument(
    "--workers", type=int, default=2, help="number of workers to load dataset"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size of the dataloader"
)
parser.add_argument(
    "--save_batch",
    type=int,
    default=1,
    help="number of samples collected before saving",
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
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--checkpoint_freq",
    type=int,
    default=10,
    help="number of epochs between two checkpoint saves",
)
parser.add_argument(
    "--burn_in", type=int, default=100, help="number of epochs before sampling"
)
parser.add_argument(
    "--sample_freq",
    type=int,
    default=100,
    help="number of epochs between two samples",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.55,
    help="gamma for learning rate scheduler",
)
args = parser.parse_args()

exp_num = check_for_exps_folders(args.save, "sgld_run")
run_filepath = os.path.join(args.save, exp_num)

os.makedirs(run_filepath, exist_ok=True)

writer = SummaryWriter(log_dir=run_filepath)


def main():
    """
    Trains a model using the SGLD optimizer.
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
        )
    )

    burnin = args.burn_in
    epochs = args.epochs
    grad_clip = args.grad_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    model = models.Network(
        C_in=image_C,
        C=6,
        H_in=image_H,
        W_in=image_W,
        num_classes=args.classes,
    ).to(device)

    optimizer = optim.SGLD(
        model.parameters(), lr=args.learning_rate, addnoise=True
    )

    gamma = args.gamma
    lr_lambda = lambda t: 1.0 / ((1 + t) ** gamma)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )

    trainer = train_utils.Trainer(
        device=device
    )
    manager = save_utils.BayesianManager(
        num_epochs=epochs,
        save_freq=10,
        num_burnin_epochs=burnin,
        save_dir=run_filepath,
        batch_size=args.save_batch,
    )

    train_accs = torch.zeros(epochs)
    train_losses = torch.zeros(epochs)
    val_accs = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = trainer.train_epoch(
            model, train_loader, criterion, optimizer, grad_clip, epoch
        )

        if epoch > burnin:
            scheduler.step()

        manager.save_sample(epoch, model)

        writer.add_scalar(f"loss/train", train_loss, epoch)
        writer.add_scalar(f"acc/train", train_acc, epoch)

        epoch_duration = time.time() - epoch_start

        writer.add_scalar(f"epoch_duration", epoch_duration, epoch)

        if manager.should_evaluate(epoch):
            val_loss, val_acc = trainer.evaluate(
                model, val_loader, criterion, epoch
            )
            writer.add_scalar(f"loss/val", val_loss, epoch)
            writer.add_scalar(f"acc/val", val_acc, epoch)

        msg = (
            "Traing: \n"
            f"Loss = {train_loss:.3f}     Accuracy = {train_acc * 100:.1f}\n"
            "\n"
            "Validation: \n"
            f"Loss = {val_loss:.3f}     Accuracy = {val_acc * 100:.1f}"
        )

        print_utils.print_msg_box(msg=msg, title=f"Epoch {epoch}")

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc
        val_losses[epoch] = val_loss
        val_accs[epoch] = val_acc

        if manager.should_save(epoch):
            manager.save_checkpoint(
                epoch,
                model,
                optimizer,
                train_losses[:epoch],
                train_accs[:epoch],
                val_losses[:epoch],
                val_accs[:epoch],
            )

    manager.finalize_sampling()

    manager.save_checkpoint(
        epoch,
        model,
        optimizer,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        f"_final",
    )

    writer.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(duration))
    print(f"The training took {formatted_time}")