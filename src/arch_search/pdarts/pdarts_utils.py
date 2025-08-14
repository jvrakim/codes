"""
Provides utility classes and functions for the PDARTS training process.

This module contains the `ArchTrainer` class, which is a modified trainer for
the architecture search phase, and the `Architect` class, which is responsible
for managing the architecture parameters and generating the final architecture.
"""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import zip_longest
from src.common.base_trainer import Trainer
from src.arch_search.pdarts.genotype import Genotype, PRIMITIVES


class ArchTrainer(Trainer):
    """
    A modified trainer for the architecture search phase.

    This trainer is responsible for training the model and the architecture
    parameters in an alternating fashion.

    Args:
        device (torch.device): The device to use for training.
    """

    def __init__(self, device):
        super().__init__(device)
        self.device = device

    def train_arch_epoch(
        self,
        model,
        train_loader,
        arch_loader,
        criterion,
        optimizer,
        arch_optimizer,
        grad_clip,
        epoch,
    ):
        """
        Trains the model and the architecture for one epoch.

        Args:
            model (torch.nn.Module): The PyTorch model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the
                training dataset.
            arch_loader (torch.utils.data.DataLoader): DataLoader for the
                architecture dataset.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer for the model
                parameters.
            arch_optimizer (torch.optim.Optimizer): The optimizer for the
                architecture parameters.
            grad_clip (float): The value to clip the gradients at.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the average loss and accuracy for the
                epoch.
        """
        model.train()

        self.loss_tracker.reset()
        self.acc_tracker.reset()

        with tqdm(train_loader, unit="batch", leave=False) as tepoch:
            for samples, arch_samples in zip_longest(tepoch, arch_loader):
                if arch_samples is None:
                    tepoch.set_description(f"Epoch {epoch}")
                    fc_input, cnn_input, target = (
                        samples["fc_input"].to(self.device),
                        samples["cnn_input"].to(self.device),
                        samples["target"].to(self.device),
                    )

                    n = fc_input.size(0)

                    loss, outputs = self.train_step(
                        model,
                        fc_input,
                        cnn_input,
                        target,
                        criterion,
                        optimizer,
                        grad_clip,
                    )
                    accuracy = self.computer.compute_accuracy(
                        outputs, target, n
                    )

                    self.loss_tracker.update(loss, n)
                    self.acc_tracker.update(accuracy, n)

                    tepoch.set_postfix(
                        {
                            "loss": self.loss_tracker.print("{:.3f}"),
                            "accuracy": self.acc_tracker.print("{:.1f}"),
                        }
                    )

                elif samples is None:
                    arch_fc_input, arch_cnn_input, arch_target = (
                        arch_samples["fc_input"].to(self.device),
                        arch_samples["cnn_input"].to(self.device),
                        arch_samples["target"].to(self.device),
                    )

                    _, _ = self.train_step(
                        model,
                        arch_fc_input,
                        arch_cnn_input,
                        arch_target,
                        criterion,
                        arch_optimizer,
                        grad_clip,
                    )

                else:
                    tepoch.set_description(f"Epoch {epoch}")
                    fc_input, cnn_input, target = (
                        samples["fc_input"].to(self.device),
                        samples["cnn_input"].to(self.device),
                        samples["target"].to(self.device),
                    )
                    arch_fc_input, arch_cnn_input, arch_target = (
                        arch_samples["fc_input"].to(self.device),
                        arch_samples["cnn_input"].to(self.device),
                        arch_samples["target"].to(self.device),
                    )

                    n = fc_input.size(0)

                    loss, outputs = self.train_step(
                        model,
                        fc_input,
                        cnn_input,
                        target,
                        criterion,
                        optimizer,
                        grad_clip,
                    )
                    _, _ = self.train_step(
                        model,
                        arch_fc_input,
                        arch_cnn_input,
                        arch_target,
                        criterion,
                        arch_optimizer,
                        grad_clip,
                    )

                    accuracy = self.computer.compute_accuracy(
                        outputs, target, n
                    )

                    self.loss_tracker.update(loss, n)
                    self.acc_tracker.update(accuracy, n)

                    tepoch.set_postfix(
                        {
                            "loss": self.loss_tracker.print("{:.3f}"),
                            "accuracy": self.acc_tracker.print("{:.1f}"),
                        }
                    )

        return self.loss_tracker.avg.item(), self.acc_tracker.avg.item()


class Architect:
    """
    Manages the architecture parameters and generates the final architecture.

    Args:
        arch_param (list of torch.Tensor): The architecture parameters.
        sm_dim (int): The dimension to apply the softmax function to.
        edges_operations_switches_normal (torch.Tensor): A tensor of booleans
            indicating which operations are active for each edge in the normal
            cells.
        edges_operations_switches_reduce (torch.Tensor): A tensor of booleans
            indicating which operations are active for each edge in the
            reduction cells.
    """

    def __init__(
        self,
        arch_param,
        sm_dim,
        edges_operations_switches_normal,
        edges_operations_switches_reduce,
    ):
        self.normal_param = arch_param[0].cpu().detach()
        self.reduce_param = arch_param[1].cpu().detach()
        self.sm_dim = sm_dim
        self.normal_prob = (
            F.softmax(self.normal_param, dim=self.sm_dim).cpu().detach()
        )
        self.reduce_prob = (
            F.softmax(self.reduce_param, dim=self.sm_dim).cpu().detach()
        )
        self.edges_operations_switches_normal = (
            edges_operations_switches_normal
        )
        self.edges_operations_switches_reduce = (
            edges_operations_switches_reduce
        )

    def drop_k_operations(self, num_edges, num_to_drop, last):
        """
        Drops the k operations with the lowest probabilities.

        Args:
            num_edges (int): The number of edges in the cell.
            num_to_drop (int): The number of operations to drop.
            last (bool): Whether this is the last time dropping operations.

        Returns:
            tuple: A tuple containing the new switches for the normal and
                reduction cells.
        """

        def _drop_k_operations(
            probs, num_to_drop, num_edges, switches, row_indices
        ):
            """
            Helper function to drop the k operations with the lowest
            probabilities.
            """
            _, drop_indices = probs.topk(
                num_to_drop, largest=False, dim=self.sm_dim
            )
            _, indices = torch.where(switches)
            indices = indices.view(num_edges, torch.sum(switches[0]))
            new_switches = switches.clone()
            new_switches[row_indices, indices[row_indices, drop_indices]] = (
                False
            )
            return new_switches

        if last:
            self.remove_zero_op()

        row_indices = np.arange(num_edges)[:, None]

        edges_operations_switches_normal = _drop_k_operations(
            self.normal_prob,
            num_to_drop,
            num_edges,
            self.edges_operations_switches_normal,
            row_indices,
        )
        edges_operations_switches_reduce = _drop_k_operations(
            self.reduce_prob,
            num_to_drop,
            num_edges,
            self.edges_operations_switches_reduce,
            row_indices,
        )

        return (
            edges_operations_switches_normal,
            edges_operations_switches_reduce,
        )

    def generate_architecture(self, nodes, max_skip_connect, num_edges):
        """
        Generates the final architecture.

        Args:
            nodes (int): The number of nodes in the cell.
            max_skip_connect (int): The maximum number of skip connections.
            num_edges (int): The number of edges in the cell.

        Returns:
            Genotype: The final architecture.
        """

        def _generate_architecture(switches, step_sizes):
            """Helper function to generate the architecture from the switches."""
            splits = switches.split(step_sizes)

            indices_list = torch.vstack(
                [torch.stack(torch.where(tensor), dim=-1) for tensor in splits]
            )

            return [
                (PRIMITIVES[indices[1].item()], indices[0].item())
                for indices in indices_list
            ]

        def _select_top_operations(
            probs, params, split_row_indices, step_sizes, k, switches
        ):
            """
            Helper function to select the top k operations for each node.
            """
            _, best_indices = probs.topk(1, dim=self.sm_dim)

            split_best_indices = best_indices.cpu().split(step_sizes)
            split_params = (
                params.detach()[row_indices, best_indices]
                .cpu()
                .split(step_sizes)
            )

            selected_indices = torch.vstack(
                [
                    split_best_indices[i][
                        F.softmax(tensor, dim=0).topk(k, dim=0)[1]
                    ].squeeze()
                    for i, tensor in enumerate(split_params)
                ]
            )

            selected_rows = torch.vstack(
                [
                    split_row_indices[i][
                        F.softmax(tensor, dim=0).topk(k, dim=0)[1]
                    ].squeeze()
                    for i, tensor in enumerate(split_params)
                ]
            )

            _, indices = torch.where(switches)
            indices = indices.view(num_edges, torch.sum(switches[0]))

            new_switches = torch.zeros_like(switches, dtype=torch.bool)
            new_switches[
                selected_rows, indices[selected_rows, selected_indices]
            ] = True

            for i in indices[selected_rows, selected_indices]:
                if i[0] == 0 or i[1] == 0:
                    print("indices", indices[selected_rows, selected_indices])
                    print("selected_indices", selected_indices)
                    print("selected_rows", selected_rows)
                    print("params", split_params)
                    print("probs", probs)

            return new_switches

        self.remove_zero_op()
        self.reduce_skip_connection_number(max_skip_connect)

        step_sizes = list(range(2, 2 + nodes))
        row_indices = torch.arange(num_edges)[:, None]
        split_row_indices = row_indices.split(step_sizes)

        edges_operations_switches_normal = _select_top_operations(
            self.normal_prob,
            self.normal_param,
            split_row_indices,
            step_sizes,
            2,
            self.edges_operations_switches_normal,
        )
        edges_operations_switches_reduce = _select_top_operations(
            self.reduce_prob,
            self.reduce_param,
            split_row_indices,
            step_sizes,
            2,
            self.edges_operations_switches_reduce,
        )

        # Parse both normal and reduce switches
        gene_normal = _generate_architecture(
            edges_operations_switches_normal, step_sizes
        )
        gene_reduce = _generate_architecture(
            edges_operations_switches_reduce, step_sizes
        )

        # Concatenation scheme
        concat = step_sizes

        # Create the genotype
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )

        return genotype

    def check_skip_conection_number(self, switches):
        """
        Checks the number of skip connections in a cell.

        Args:
            switches (torch.Tensor): A tensor of booleans indicating which
                operations are active for each edge.

        Returns:
            int: The number of skip connections.
        """
        return len(torch.where(switches[:, 3])[0])

    def reduce_skip_connection_number(self, max_skip_connect):
        """
        Reduces the number of skip connections in a cell.

        Args:
            max_skip_connect (int): The maximum number of skip connections.
        """
        skip_connection_rows = torch.where(
            self.edges_operations_switches_normal[:, 3]
        )[0]
        skip_connection_column = torch.sum(
            self.edges_operations_switches_normal[skip_connection_rows, :3],
            dim=1,
        )

        skip_connection_probs = F.softmax(
            self.normal_param[skip_connection_rows, skip_connection_column],
            dim=0,
        )

        skip_connection_num = self.check_skip_conection_number(
            self.edges_operations_switches_normal
        )

        if skip_connection_num > max_skip_connect:
            _, drop_indices = skip_connection_probs.topk(
                int(skip_connection_num - max_skip_connect),
                largest=False,
                dim=self.sm_dim,
            )

            self.normal_prob[
                skip_connection_rows[drop_indices],
                skip_connection_column[drop_indices],
            ] = -1

    def remove_zero_op(self):
        """Removes the 'none' operation from the search space."""
        zero_op_indices_normal = torch.where(
            self.edges_operations_switches_normal[:, 0]
        )[0]
        zero_op_indices_reduce = torch.where(
            self.edges_operations_switches_reduce[:, 0]
        )[0]
        self.normal_prob[zero_op_indices_normal, 0] = -1
        self.reduce_prob[zero_op_indices_reduce, 0] = -1


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)