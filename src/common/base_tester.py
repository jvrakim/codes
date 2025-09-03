"""
This module defines the base class for all testers.
"""

import os
from abc import ABC, abstractmethod

import torch


class BaseTester(ABC):
    """
    Abstract base class for testers.
    """

    def __init__(self, device, save_path):
        self.device = device
        self.save_path = save_path

    @abstractmethod
    def test(self, model, test_loader, *args, **kwargs):
        """
        This method should be implemented by the subclasses.
        """
        raise NotImplementedError

    def save_results(self, results, filename="test_results.pt"):
        """
        Saves the results of the testing to a file.
        """
        torch.save(results, os.path.join(self.save_path, filename))