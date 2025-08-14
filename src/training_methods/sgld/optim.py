"""
Provides the SGLD optimizer.

This module contains the `SGLD` optimizer, which is a stochastic gradient
Langevin dynamics optimizer based on PyTorch's SGD.
"""

import torch
from torch.optim.optimizer import Optimizer, required


class SGLD(Optimizer):
    """
    An SGLD optimizer based on PyTorch's SGD.

    This optimizer implements the stochastic gradient Langevin dynamics
    algorithm, which is a stochastic gradient descent algorithm that adds
    Gaussian noise to the parameter updates.

    Args:
        params (iterable): An iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float): The learning rate.
        addnoise (bool, optional): Whether to add noise to the parameter
            updates. Defaults to True.
    """

    def __init__(self, params, lr=required, addnoise=True):
        weight_decay = 0

        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)

        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group["addnoise"]:
                    langevin_noise = (
                        1e-2
                        * p.data.new(p.data.size()).normal_(mean=0, std=1)
                        / torch.sqrt(torch.tensor(group["lr"]))
                    )
                    noised_grad = torch.add(0.5 * d_p, langevin_noise)
                    p.data.add_(noised_grad, alpha=-group["lr"])
                else:
                    p.data.add_(-group["lr"], 0.5 * d_p)

        return loss