"""
This script provides a unified interface for running posterior network-based
neural network experiments.

It supports training, testing, and running a full experiment (both training and
testing) with a single command.
"""

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.common import data_utils
from src.common import print_utils
from src.common import base_saver as save_utils
from src.training_methods.posterior import train_utils
from src.training_methods.posterior.parser_utils import PosteriorParser
from src.training_methods.posterior.posterior_network import PosteriorNetwork

def setup_experiment(args):
    """
    Set up the experiment environment.

    This function prepares the data loaders, device, model, and criterion
    based on the experiment type and objective.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        dict: A dictionary containing the setup components:
              'device', 'train_loader', 'val_loader', 'test_loader',
              'model', 'criterion', and other experiment-specific data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_kwargs = {
        "return_dimensions": True,
        "return_class_counts": True
    }
    
    (
        train_loader,
        val_loader,
        test_loader,
        *other_info,
    ) = data_utils.get_data_loaders(
        data_path=args.path_to_data,
        training_dataset=args.training_dataset,
        testing_dataset=args.test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_classes=args.classes,
        **dataloader_kwargs,
    )

    image_dims = other_info[0]
    image_C, image_H, image_W = image_dims

    results = {
        "device": device,
        "train_loader": None,
        "val_loader": None,
        "test_loader": None,
        "model": None,
        "criterion": None,
    }

    if args.mode == "train":
        results["train_loader"] = train_loader
        results["val_loader"] = val_loader
    elif args.mode == "test":
        results["test_loader"] = test_loader
    elif args.mode == "full":
        results["train_loader"] = train_loader
        results["val_loader"] = val_loader
        results["test_loader"] = test_loader

    
    N = other_info[1]

    results["model"] = PosteriorNetwork(
            N=N,
            C_in=image_C,
            H_in=image_H,
            W_in=image_W,
            C=6,
            output_dim=args.classes,
            latent_dim=4,
            n_density=4,
            seed=args.seed,
        ).to(device)
    
    results["criterion"] = train_utils.uce_loss
    results["N"] = N

    return results

def train(args, setup):
    """
    Trains a posterior network model.

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
    experiment_name = f"posterior_{args.training_dataset}_{timestamp}"
    experiment_path = os.path.join(args.experiment_path, experiment_name)

    training_files_path = os.path.join(experiment_path, "training_files")
    checkpoints_path = os.path.join(training_files_path, "checkpoints")
    best_model_path = os.path.join(training_files_path, "best_model")
    logs_path = os.path.join(training_files_path, "logs")

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_path)

    device = setup["device"]
    train_loader = setup["train_loader"]
    val_loader = setup["val_loader"]
    model = setup["model"]
    criterion = setup["criterion"]

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    trainer = train_utils.PosteriorTrainer(device, args.classes, args.regr)
    manager = save_utils.Manager(
        num_epochs=args.epochs,
        save_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
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

    writer.close()
    print(f"Finished training. Best model saved in {best_model_path}")


def test(args, setup):
    """
    Tests a posterior network model.

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

    device = setup["device"]
    test_loader = setup["test_loader"]
    model = setup["model"]
    criterion = setup["criterion"]

    path = os.path.join(best_model_path, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = train_utils.uce_loss
    trainer = train_utils.PosteriorTrainer(device, args.classes, args.regr)

    targets, alphas, probs, ids = trainer.evaluate(
        model, test_loader, criterion, 0, return_outputs=True
    )

    alphas = {
        "ids": ids,
        "alphas": alphas,
        "probs": probs,
        "targets": targets,
    }

    torch.save(alphas, os.path.join(test_path, "test_results.pt"))
    print(f"Finished testing. Test results saved in {test_path}")


def main():
    """
    The main function of the script.

    It parses the command-line arguments and calls the appropriate
    function based on the specified mode.
    """
    parser = PosteriorParser()
    args = parser.parse_args()

    setup = setup_experiment(args)

    if args.mode == "train":
        train(args, setup)
    elif args.mode == "test":
        test(args, setup)
    elif args.mode == "full":
        train(args, setup)
        test(args, setup)
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
