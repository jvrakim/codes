"""
This script runs the analysis for the Laplace, Ensemble, and Posterior models.

It loads the trained models, runs the analysis for each model, and then plots
the combined results.
"""

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Dict, Any

# Laplace imports
from laplace import Laplace
from src.training_methods.laplace import collate
from src.training_methods.laplace.metric_utils import LaplaceMetric
from src.training_methods.laplace.analysis_utils import LaplaceAnalyzer
from src.training_methods.laplace.model import LaplaceModel

# Ensemble imports
from src.training_methods.ensemble import save_utils as ensemble_save_utils
from src.training_methods.ensemble import train_utils as ensemble_train_utils
from src.training_methods.ensemble.metric_utils import EnsembleMetric
from src.training_methods.ensemble.analysis_utils import EnsembleAnalyzer

# Posterior imports
from src.training_methods.posterior.posterior_network import PosteriorNetwork
from src.training_methods.posterior import train_utils as posterior_train_utils
from src.training_methods.posterior.metric_utils import PosteriorMetric
from src.training_methods.posterior.analysis_utils import PosteriorAnalyzer

# Common imports
from src.common import data_utils
from src.common.base_model import Network

# For plotting
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument(
        "--data_path",
        type=str,
        default="/home/user/data/processed",
        help="path to datasets",
    )
parser.add_argument(
        "--results_path",
        type=str,
        default="/home/user/results",
        help="path to save images",
    )
args = parser.parse_args()


def run_laplace_analysis(
    device: torch.device, run_filepath: str, data_path: str, results_path: str
) -> LaplaceAnalyzer:
    """
    Runs the analysis for the Laplace model.

    Args:
        device (torch.device): The device to run the analysis on.
        run_filepath (str): The path to the directory containing the trained
            Laplace model.
        data_path (str): The path to the data.
        results_path (str): The path to save the results to.

    Returns:
        LaplaceAnalyzer: An analyzer object containing the results of the
            analysis.
    """
    print("\n--- Running Laplace Analysis ---")

    train_loader, val_loader, test_loader, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path=data_path,
            batch_size=256,
            num_workers=15,
            num_classes=2,
            return_dimensions=True,
            return_tuple=True,
            collate_fn=collate.custom_collate_fn,
        )
    )

    model = LaplaceModel(
        C_in=image_C,
        C=6,
        H_in=image_H,
        W_in=image_W,
        num_classes=2,
    ).to(device)

    path = os.path.join(run_filepath, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="full",
    )
    la.fit(train_loader)

    la.optimize_prior_precision(
        method="gridsearch",
        pred_type="glm",
        link_approx="mc",
        val_loader=val_loader,
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


    la_metric = LaplaceMetric(device)
    laplace_results = la_metric.get_metrics(samples=predictions, target=targets)

    laplace_analyzer_obj = LaplaceAnalyzer(laplace_results)
    print("Laplace Analysis Complete.")
    return laplace_analyzer_obj


def run_ensemble_analysis(
    device: torch.device, run_filepath: str, data_path: str, results_path: str
) -> EnsembleAnalyzer:
    """
    Runs the analysis for the Ensemble model.

    Args:
        device (torch.device): The device to run the analysis on.
        run_filepath (str): The path to the directory containing the trained
            Ensemble model.
        data_path (str): The path to the data.
        results_path (str): The path to save the results to.

    Returns:
        EnsembleAnalyzer: An analyzer object containing the results of the
            analysis.
    """
    print("\n--- Running Ensemble Analysis ---")
    num_models = 10
    train_loader, val_loader, test_loader, (image_C, image_H, image_W) = (
        data_utils.get_data_loaders(
            data_path=data_path,
            batch_size=256,
            num_workers=15,
            num_classes=2,
            return_dimensions=True,
        )
    )

    model_list = [
        Network(
            C_in=image_C,
            C=6,
            H_in=image_H,
            W_in=image_W,
            num_classes=2,
        ).to(device)
        for _ in range(num_models)
    ]
    base_model_instance = copy.deepcopy(model_list[0])
    base_model_instance = base_model_instance.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    model_list = ensemble_save_utils.load_ensemble_models(
        run_filepath, model_list
    )
    params, buffers = torch.func.stack_module_state(model_list)
    trainer = ensemble_train_utils.EnsembleTrainer(
        device, num_models, params, buffers
    )
    targets, outputs, ids = trainer.evaluate(
        base_model_instance, test_loader, criterion, 0, return_outputs=True
    )
    probs = F.softmax(outputs.mean(dim=0), dim=1)
    individual_probs = F.softmax(outputs, dim=2)

    e_metric = EnsembleMetric(device=device)
    ensemble_results = e_metric.get_metrics(
        probs, individual_probs, ids, targets
    )

    ensemble_analyzer_obj = EnsembleAnalyzer(ensemble_results)
    print("Ensemble Analysis Complete.")
    return ensemble_analyzer_obj


def run_posterior_analysis(
    device: torch.device, run_filepath: str, data_path: str, results_path: str
) -> PosteriorAnalyzer:
    """
    Runs the analysis for the Posterior model.

    Args:
        device (torch.device): The device to run the analysis on.
        run_filepath (str): The path to the directory containing the trained
            Posterior model.
        data_path (str): The path to the data.
        results_path (str): The path to save the results to.

    Returns:
        PosteriorAnalyzer: An analyzer object containing the results of the
            analysis.
    """
    print("\n--- Running Posterior Analysis ---")
    train_loader, val_loader, test_loader, (image_C, image_H, image_W), N = (
        data_utils.get_data_loaders(
            data_path=data_path,
            batch_size=256,
            num_workers=15,
            num_classes=2,
            return_dimensions=True,
            return_class_counts=True,
        )
    )

    model = PosteriorNetwork(
        N=N,
        C_in=image_C,
        H_in=image_H,
        W_in=image_W,
        C=6,
        output_dim=2,
        latent_dim=4,
        n_density=4,
        seed=123,
    ).to(device)

    criterion = posterior_train_utils.uce_loss
    post_trainer = posterior_train_utils.PosteriorTrainer(device, 2, 1e-5)

    path = os.path.join(run_filepath, "model_best.pt")
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, alphas, probs, ids = post_trainer.evaluate(
        model, test_loader, criterion, 0, return_outputs=True
    )

    post_metric = PosteriorMetric(device)
    post_results = post_metric.get_metrics(alphas, ids, targets)

    posterior_analyzer_obj = PosteriorAnalyzer(post_results)
    print("Posterior Analysis Complete.")
    return posterior_analyzer_obj


def plot_combined_results(analyzers: Dict[str, Any], results_path: str):
    """
    Plots the combined results of the analysis.

    Args:
        analyzers (dict): A dictionary of analyzer objects, where the keys are
            the model names and the values are the analyzer objects.
        results_path (str): The path to save the plots to.
    """
    colors = ["#0077BB", "#EE7733", "#009988"]
    alpha = 0.9
    fontsize = 18
    figsize = (10, 6)
    model_names = list(analyzers.keys())

    # Calibration Curve
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(111)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

    n_bins = 20
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    total_bar_width = 0.8 * (1.0 / n_bins)
    individual_bar_width = total_bar_width / len(model_names)

    for i, (name, analyzer_obj) in enumerate(analyzers.items()):
        accuracies, confidences_in_bins = analyzer_obj.get_calibration_curve(
            n_bins=n_bins
        )
        offset = (i - len(model_names) / 2 + 0.5) * individual_bar_width
        ax1.bar(
            bin_centers + offset,
            accuracies,
            width=individual_bar_width,
            color=colors[i],
            alpha=alpha,
            edgecolor="black",
            linewidth=0.5,
            label=name,
        )

    ax1.set_xticks(bin_boundaries[::2])
    ax1.set_xticklabels(
        [f"{x:.2f}" for x in bin_boundaries[::2]], rotation=45, ha="right"
    )
    ax1.set_xticks(bin_boundaries[1::2], minor=True)
    ax1.set_xlabel("Confidence", fontsize=fontsize + 3)
    ax1.set_ylabel("Accuracy", fontsize=fontsize + 3)
    ax1.set_title("Calibration Curve", fontsize=fontsize + 4)
    ax1.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_path, "combined_calibration_curve.pdf"),
        dpi=150,
        format="pdf",
    )
    plt.close()

    # ROC Curve
    plt.figure(figsize=figsize)
    for i, (name, analyzer_obj) in enumerate(analyzers.items()):
        fpr, tpr, roc_auc = analyzer_obj.get_roc_curve()
        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=3,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=3,
        linestyle="--",
        label="Random Classifier",
    )
    plt.plot(
        [0, 0, 1],
        [0, 1, 1],
        color="green",
        lw=3,
        linestyle=":",
        label="Perfect Classifier",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=fontsize + 3)
    plt.ylabel("True Positive Rate (TPR)", fontsize=fontsize + 3)
    plt.xticks(fontsize=fontsize + 2)
    plt.yticks(fontsize=fontsize + 2)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=fontsize + 4)
    plt.legend(fontsize=fontsize, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_path, "combined_roc_curve.pdf"),
        dpi=150,
        format="pdf",
    )
    plt.close()

    # Risk-Coverage Curve
    plt.figure(figsize=figsize)
    for i, (name, analyzer_obj) in enumerate(analyzers.items()):
        coverages, risks, rc_auc = analyzer_obj.get_risk_coverage_curve()
        plt.plot(
            coverages,
            risks,
            color=colors[i],
            lw=3,
            label=f"{name} (AUC = {rc_auc:.3f})",
        )

    plt.xlabel("Coverage", fontsize=fontsize + 3)
    plt.ylabel("Risk", fontsize=fontsize + 3)
    plt.title("Risk-Coverage Curve", fontsize=fontsize + 4)
    plt.xticks(fontsize=fontsize + 2)
    plt.yticks(fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_path, "combined_risk_coverage_curve.pdf"),
        dpi=150,
        format="pdf",
    )
    plt.close()

    # Print metrics
    for name, analyzer_obj in analyzers.items():
        print(f"\n{name} Metrics:")
        for key, value in analyzer_obj.results.items():
            if isinstance(value, (float, int, np.float32, np.float64)):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    laplace_run_filepath = "/home/joao/auger/codes/results/networks/laplace_run01"
    ensemble_run_filepath = "/home/joao/auger/codes/results/networks/ensemble_run01"
    posterior_run_filepath = "/home/joao/auger/codes/results/networks/posterior_run02"

    analyzers = {}

    analyzers["Laplace"] = run_laplace_analysis(
        device, laplace_run_filepath, args.data_path, args.results_path
    )

    analyzers["Ensemble"] = run_ensemble_analysis(
        device, ensemble_run_filepath, args.data_path, args.results_path
    )
    analyzers["Posterior"] = run_posterior_analysis(
        device, posterior_run_filepath, args.data_path, args.results_path
    )

    plot_combined_results(analyzers, args.results_path)
