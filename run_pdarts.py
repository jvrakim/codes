"""
This script runs the PDARTS architecture search.

The search is divided into stages, where each stage increases the depth of the
network and prunes the search space.
"""

import os
import sys
import time
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.common import print_utils
from src.arch_search.pdarts import pdarts_utils
from src.arch_search.pdarts.model import Network
from src.common import base_saver as save_utils
from src.common import data_utils
from src.arch_search.pdarts.genotype import PRIMITIVES
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--num_of_stages",
    default=3,
    type=int,
    help="number of search stages. With each stage the network becomes deeper",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./",
    help="path for the data to be used duing search",
)
parser.add_argument(
    "--workers", type=int, default=2, help="number of workers to load dataset"
)
parser.add_argument("--batch_size", type=int, default=96, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--betas", type=float, nargs=2, default=[0.9, 0.999], help="Adam betas"
)
parser.add_argument(
    "--weight_decay", type=float, default=3e-4, help="weight decay"
)
parser.add_argument(
    "--epochs", type=int, default=25, help="num of training epochs"
)
parser.add_argument(
    "--classes",
    type=int,
    default=2,
    help="number of classes to be predicted by the network",
)
parser.add_argument(
    "--layers",
    nargs="+",
    type=int,
    default=[5, 11, 17],
    help="total number of layers. Must have num_of_stages length",
)
parser.add_argument(
    "--save", type=str, default="./", help="folder to save the experiments"
)
parser.add_argument(
    "--save_freq", type=int, default=10, help="saving frequency"
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
    "--arch_learning_rate",
    type=float,
    default=6e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
parser.add_argument(
    "--dropout_rate",
    nargs="+",
    type=float,
    default=[0.0, 0.3, 0.6],
    help="dropout rate of skip connect. Must have num_of_stages length",
)
parser.add_argument(
    "--add_width",
    nargs="+",
    type=int,
    default=[0, 0, 0],
    help="add channels. Must have num_of_stages length",
)
parser.add_argument(
    "--add_layers",
    nargs="+",
    type=int,
    default=[0, 0, 0],
    help="add layers. Must have num_of_stages length",
)
parser.add_argument(
    "--num_to_keep",
    nargs="+",
    type=int,
    default=[5, 3, 1],
    help="number of edges to keep. Must have num_of_stages length",
)
parser.add_argument(
    "--num_to_drop",
    nargs="+",
    type=int,
    default=[3, 2, 2],
    help="number of edges to drop. Must have num_of_stages length",
)
parser.add_argument(
    "--eps_no_archs",
    nargs="+",
    type=int,
    default=[10, 10, 10],
    help="number of epochs that only the weights are trained. Must have num_of_stages length",
)
parser.add_argument(
    "--max_skip_connect",
    type=int,
    default=2,
    help="number of skip connections allowed at the end of the search",
)
parser.add_argument(
    "--scale_factor",
    type=float,
    default=0.2,
    help="scale factor of the dropout probability",
)
parser.add_argument(
    "--num_nodes",
    type=int,
    default=4,
    help="number of nodes inside the graph, equivalent to the number of neurons inside of the cell",
)
parser.add_argument(
    "--checkpoint_freq",
    type=int,
    default=10,
    help="number of epochs between two checkpoint saves",
)

args = parser.parse_args()

assert len(args.layers) == args.num_of_stages
assert len(args.add_width) == args.num_of_stages
assert len(args.add_layers) == args.num_of_stages
assert len(args.num_to_keep) == args.num_of_stages
assert len(args.num_to_drop) == args.num_of_stages
assert len(args.eps_no_archs) == args.num_of_stages
assert len(args.dropout_rate) == args.num_of_stages

exp_num = save_utils.check_for_exps_folders(args.save, "pdarts_run")
run_filepath = os.path.join(args.save, exp_num)

os.makedirs(run_filepath, exist_ok = True)

writer = SummaryWriter(log_dir = run_filepath)


def main():
    """
    Runs the PDARTS architecture search.
    """
    if not torch.cuda.is_available():
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    train_loader, arch_loader, _, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path = args.data_path,
            batch_size = args.batch_size,
            num_workers = args.workers,
            train_portion = args.train_portion,
            return_dimensions = True,
        )
    )

    num_to_keep = args.num_to_keep
    num_to_drop = args.num_to_drop
    eps_no_archs = args.eps_no_archs
    drop_rate = args.dropout_rate
    sm_dim = -1
    epochs = args.epochs
    scale_factor = args.scale_factor
    grad_clip = args.grad_clip
    max_skip_connect = args.max_skip_connect

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    trainer = pdarts_utils.ArchTrainer(device)
    manager = save_utils.Manager(
        num_epochs=epochs,
        save_freq=args.save_freq,
        save_dir=run_filepath,
    )

    nodes = args.num_nodes
    num_edges = int(nodes * (nodes + 3) / 2)
    num_op = int(len(PRIMITIVES))

    edges_operations_switches_normal = torch.ones(
        (num_edges, num_op), dtype=torch.bool
    )
    edges_operations_switches_reduce = torch.ones(
        (num_edges, num_op), dtype=torch.bool
    )

    train_accs = torch.zeros(epochs)
    train_losses = torch.zeros(epochs)
    val_accs = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)

    # Main architecture search loop
    for stage in range(args.num_of_stages):
        print(f"Creating Network at stage {stage}")
        model = Network(
            image_C + args.add_width[stage],
            args.classes,
            args.layers[stage] + args.add_layers[stage],
            edges_operations_switches_normal=edges_operations_switches_normal,
            edges_operations_switches_reduce=edges_operations_switches_reduce,
            p=float(drop_rate[stage]),
            H_in=image_H,
            W_in=image_W,
        )

        model = model.to(device)

        network_params = nn.ParameterList(
            [
                v
                for k, v in model.named_parameters()
                if not (
                    k.endswith("alphas_normal") or k.endswith("alphas_reduce")
                )
            ]
        )

        optimizer = torch.optim.Adam(
            network_params,
            lr=args.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )

        arch_optimizer = torch.optim.Adam(
            model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay,
        )

        eps_no_arch = eps_no_archs[stage]

        for epoch in range(epochs):
            epoch_start = time.time()

            if epoch < eps_no_arch:
                model.p = (
                    float(drop_rate[stage]) * (epochs - epoch - 1) / epochs
                )
                model.update_p()
                train_loss, train_acc = trainer.train_epoch(
                    model, train_loader, criterion, optimizer, grad_clip, epoch
                )

            else:
                model.p = float(drop_rate[stage]) * np.exp(
                    -(epoch - eps_no_arch) * scale_factor
                )
                model.update_p()
                train_loss, train_acc = trainer.train_arch_epoch(
                    model,
                    train_loader,
                    arch_loader,
                    criterion,
                    optimizer,
                    arch_optimizer,
                    grad_clip,
                    epoch,
                )

            writer.add_scalar(f"loss/train, stage{stage}", train_loss, epoch)
            writer.add_scalar(f"acc/train, stage{stage}", train_acc, epoch)

            epoch_duration = time.time() - epoch_start

            writer.add_scalar(
                f"epoch_duration, stage{stage}", epoch_duration, epoch
            )

            if manager.should_evaluate(epoch):
                val_loss, val_acc = trainer.evaluate(
                    model, arch_loader, criterion, epoch
                )
                writer.add_scalar(f"loss/val, stage{stage}", val_loss, epoch)
                writer.add_scalar(f"acc/val, stage{stage}", val_acc, epoch)

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
                    stage,
                )

            msg = (
                "Traing: \n"
                f"Loss = {train_loss:.3f}     Accuracy = {train_acc * 100:.1f}\n"
                "\n"
                "Validation: \n"
                f"Loss = {val_loss:.3f}     Accuracy = {val_acc * 100:.1f}"
            )

            print_utils.print_msg_box(msg=msg, title=f"Epoch {epoch}")

        manager.save_checkpoint(
            epoch,
            model,
            optimizer,
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            f"end_of_stage_{stage}",
        )

        arch_param = model.arch_parameters()
        architect = pdarts_utils.Architect(
            arch_param,
            sm_dim,
            edges_operations_switches_normal,
            edges_operations_switches_reduce,
        )

        last = stage == len(num_to_keep) - 1

        edges_operations_switches_normal, edges_operations_switches_reduce = (
            architect.drop_k_operations(num_edges, num_to_drop[stage], last)
        )

        best_architecture = architect.generate_architecture(
            nodes, max_skip_connect, num_edges
        )

        torch.save(
            best_architecture,
            os.path.join(run_filepath, f"best_architecture_{stage}.pt"),
        )

        torch.save(
            edges_operations_switches_normal,
            os.path.join(run_filepath, f"switches_normal_{stage}.pt"),
        )
        torch.save(
            edges_operations_switches_reduce,
            os.path.join(run_filepath, f"switches_reduce_{stage}.pt"),
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f"The search took {duration}")