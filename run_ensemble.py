"""
This script trains an ensemble of models.

It uses the `torch.func` library to train an ensemble of models in parallel.
"""

import os
import sys
import time
import copy
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from src.training_methods.ensemble import save_utils
from src.training_methods.ensemble import train_utils
from src.common import data_utils
from src.common import print_utils
from src.common.base_model import Network
from torch.func import stack_module_state
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
# common arguments
parser.add_argument(
    "--data_path",
    type=str,
    default="./",
    help="path for the data to be used duing search",
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
# specifc arguments
parser.add_argument(
    "--num_models", type=int, default=10, help="number of models for ensemble"
)

args = parser.parse_args()

exp_num = save_utils.check_for_exps_folders(args.save, "ensemble_run")
run_filepath = os.path.join(args.save, exp_num)

os.makedirs(run_filepath, exist_ok=True)

writer = SummaryWriter(log_dir=run_filepath)


def main():
    """
    Trains an ensemble of models.
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

    epochs = args.epochs
    grad_clip = args.grad_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_models = args.num_models
    model_list = [
        Network(
            C_in=image_C,
            C=6,
            H_in=image_H,
            W_in=image_W,
            num_classes=args.classes,
        ).to(device)
        for _ in range(num_models)
    ]

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    params, buffers = stack_module_state(model_list)

    base_model = copy.deepcopy(model_list[0])
    base_model = base_model.to(device)

    optimizer = optim.SGD(list(params.values()), lr=args.learning_rate)

    trainer = train_utils.EnsembleTrainer(device, num_models, params, buffers)

    manager = save_utils.EnsembleManager(
        num_epochs=epochs,
        save_freq=args.checkpoint_freq,
        eval_freq=1,
        save_dir=run_filepath,
        filename_prefix=None,
        num_models=num_models,
    )

    train_accs = torch.zeros(epochs, num_models)
    train_losses = torch.zeros(epochs, num_models)
    num_eval_points = (epochs + args.eval_freq - 1) // args.eval_freq
    val_accs = torch.zeros(num_eval_points, num_models)
    val_losses = torch.zeros(num_eval_points, num_models)

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = trainer.train_epoch(
            base_model, train_loader, criterion, optimizer, grad_clip, epoch
        )

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        for i in range(num_models):
            writer.add_scalar(f"model_{i}_loss/train", train_loss[i], epoch)
            writer.add_scalar(f"model_{i}_acc/train", train_acc[i], epoch)

        epoch_duration = time.time() - epoch_start

        writer.add_scalar("epoch_duration", epoch_duration, epoch)

        if manager.should_evaluate(epoch):
            val_loss, val_acc = trainer.evaluate(
                base_model, val_loader, criterion, epoch
            )

            eval_idx = epoch // args.eval_freq
            val_losses[eval_idx] = val_loss
            val_accs[eval_idx] = val_acc

            for i in range(num_models):
                writer.add_scalar(f"model_{i}_loss/val", val_loss[i], epoch)
                writer.add_scalar(f"model_{i}_acc/val", val_acc[i], epoch)

            manager.save_best(
                epoch,
                params,
                buffers,
                optimizer,
                train_losses[: epoch + 1],
                train_accs[: epoch + 1],
                val_losses[: eval_idx + 1],
                val_accs[: eval_idx + 1],
            )

        loss_fmt = "{:.3f}"
        acc_fmt = "{:.1f}"

        train_loss_str = [loss_fmt.format(loss) for loss in train_loss]
        train_acc_str = [acc_fmt.format(acc * 100) for acc in train_acc]
        val_loss_str = [loss_fmt.format(loss) for loss in val_loss]
        val_acc_str = [acc_fmt.format(acc * 100) for acc in val_acc]

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
                params,
                buffers,
                optimizer,
                train_losses[: epoch + 1],
                train_accs[: epoch + 1],
                val_losses[: eval_idx + 1],
                val_accs[: eval_idx + 1],
            )

    writer.close()

    model_list = save_utils.load_ensemble_models(run_filepath, model_list)
    params, buffers = stack_module_state(model_list)

    predictor = train_utils.EnsembleTrainer(
        device, num_models, params, buffers
    )
    targets, outputs, ids = predictor.evaluate(
        base_model, test_loader, criterion, 0, return_outputs=True
    )
    probs = F.softmax(outputs.mean(dim=0), dim=1)
    individual_probs = F.softmax(outputs, dim=2)

    probs = {
        "ids": ids,
        "probs": probs,
        "targets": targets,
        "individual_probs": individual_probs,
    }

    torch.save(probs, os.path.join(run_filepath, "probs.pt"))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(duration))
    print(f"The training took {formatted_time}")
