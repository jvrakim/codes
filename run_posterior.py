"""
This script trains a posterior network model.
"""

import os
import sys
import time
import torch
import argparse

import torch.optim as optim
import torch.backends.cudnn as cudnn

from src.training_methods.posterior.posterior_network import PosteriorNetwork
from src.common import base_saver as save_utils
from src.training_methods.posterior import train_utils
from src.common import data_utils
from src.common import print_utils
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
# common arguments
parser.add_argument(
    "--data_path",
    type=str,
    default="./",
    help="path for the data to be used during training",
)
parser.add_argument(
    "--workers", type=int, default=2, help="number of workers to load dataset"
)
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
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
    "--regr",
    type=float,
    default=0.01,
    help="regularization parameter for posterior network",
)

args = parser.parse_args()

exp_num = save_utils.check_for_exps_folders(args.save, "posterior_run")
run_filepath = os.path.join(args.save, exp_num)

os.makedirs(run_filepath, exist_ok=True)

writer = SummaryWriter(log_dir=run_filepath)


def main():
    """
    Trains a posterior network model.
    """
    if not torch.cuda.is_available():
        sys.exit(1)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader, (image_C, image_H, image_W), N = (
        data_utils.get_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.workers,
            num_classes=args.classes,
            return_dimensions=True,
            return_class_counts=True,
        )
    )

    epochs = args.epochs
    grad_clip = args.grad_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PosteriorNetwork(
        N=N,
        C_in=image_C,
        H_in=image_H,
        W_in=image_W,
        C=6,
        output_dim=args.classes,
        latent_dim=4,
        n_density=4,
        seed=123,
    ).to(device)

    criterion = train_utils.uce_loss

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    trainer = train_utils.PosteriorTrainer(device, args.classes, args.regr)
    manager = save_utils.Manager(
        num_epochs=epochs,
        save_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        save_dir=run_filepath,
        filename_prefix=None,
    )

    train_accs = torch.zeros(epochs)
    train_losses = torch.zeros(epochs)
    num_eval_points = (epochs + args.eval_freq - 1) // args.eval_freq
    val_accs = torch.zeros(num_eval_points)
    val_losses = torch.zeros(num_eval_points)

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = trainer.train_epoch(
            model, train_loader, criterion, optimizer, grad_clip, epoch
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

    writer.close()

    path = os.path.join(run_filepath, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, alphas, probs, ids = trainer.evaluate(
        model, test_loader, criterion, 0, return_outputs=True
    )

    alphas = {
        "ids": ids,
        "alphas": alphas,
        "probs": probs,
        "targets": targets,
    }

    torch.save(alphas, os.path.join(run_filepath, "alphas.pt"))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(duration))
    print(f"The training took {formatted_time}")