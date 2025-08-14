"""
Provides utility functions and classes for handling data.

This module contains functions for creating data loaders, splitting datasets,
and computing statistics. It also includes the `CustomDataset` and `MultiInput`
classes for handling the specific data format of this project.
"""

import torch
import os
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Tuple, Dict, Any, Optional


def get_class_counts(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Counts the number of instances for each class in a dataset.

    Args:
        dataset (Dataset): The dataset to count the class instances of.
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: A tensor containing the counts for each class.
    """
    if isinstance(dataset, CustomDataset):
        targets = dataset.data_dict["primary_array"]
    else:
        targets = torch.tensor(
            [dataset[i]["target"] for i in range(len(dataset))]
        )

    classes, counts = torch.unique(targets, return_counts=True)
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    class_counts[classes.long()] = counts

    return class_counts


def split_dataset(
    dataset: Dataset, train_portion: float
) -> List[Dataset]:
    """
    Splits a dataset into two parts.

    Args:
        dataset (Dataset): The dataset to split.
        train_portion (float): The proportion of the dataset to use for the
            first split.

    Returns:
        list: A list containing the two dataset splits.
    """
    return random_split(dataset, [train_portion, 1 - train_portion])


def compute_mean_std(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mean and standard deviation of a tensor.

    Args:
        array (torch.Tensor): The input tensor.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the
            tensor.
    """
    array_mean = torch.mean(array)
    array_std = torch.std(array)
    array_std[array_std == 0] = 1.0
    return array_mean, array_std


def standardize_data(
    data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Standardizes a tensor using the given mean and standard deviation.

    Args:
        data (torch.Tensor): The tensor to standardize.
        mean (torch.Tensor): The mean to use for standardization.
        std (torch.Tensor): The standard deviation to use for standardization.

    Returns:
        torch.Tensor: The standardized tensor.
    """
    return data.sub(mean).div(std)


def get_dimensions(data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gets the dimensions of the shower plane array in a data dictionary.

    Args:
        data_dict (dict): The data dictionary.

    Returns:
        tuple: A tuple containing the number of channels, height, and width
            of the shower plane array.
    """
    C = torch.tensor(data_dict["shower_plane_array"].shape[1])
    H = torch.tensor(data_dict["shower_plane_array"].shape[2])
    W = torch.tensor(data_dict["shower_plane_array"].shape[3])
    return C, H, W


def get_data_loaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    return_dimensions: bool = False,
    return_class_counts: bool = False,
    return_tuple: bool = False,
    collate_fn: Optional[callable] = None,
) -> tuple:
    """
    Creates train, validation, and test data loaders.

    Args:
        data_path (str): The path to the folder containing the data files.
        batch_size (int): The batch size for the data loaders.
        num_workers (int): The number of workers for data loading.
        num_classes (int): The number of classes in the dataset.
        return_dimensions (bool, optional): Whether to return the image
            dimensions. Defaults to False.
        return_class_counts (bool, optional): Whether to return the class
            counts. Defaults to False.
        return_tuple (bool, optional): Whether to return the data as a tuple
            of (inputs, target) or a dictionary. Defaults to False.
        collate_fn (callable, optional): The collate function to use for the
            data loaders. Defaults to None.

    Returns:
        tuple: A tuple containing the train, validation, and test data
            loaders, and optionally the image dimensions and class counts.
    """
    train_path = os.path.join(data_path, "train_data.pt")
    val_path = os.path.join(data_path, "val_data.pt")
    test_path = os.path.join(data_path, "test_data.pt")

    train_dict = torch.load(train_path, weights_only=False)
    val_dict = torch.load(val_path, weights_only=False)
    test_dict = torch.load(test_path, weights_only=False)

    train_dataset = CustomDataset(train_dict, num_classes, return_tuple)
    val_dataset = CustomDataset(val_dict, num_classes, return_tuple)
    test_dataset = CustomDataset(test_dict, num_classes, False)

    train_loader = DataLoader(
        train_dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return_values = [train_loader, val_loader, test_loader]

    if return_dimensions:
        dimensions = get_dimensions(train_dataset.data_dict)
        return_values.append(dimensions)

    if return_class_counts:
        class_counts = get_class_counts(train_dataset, num_classes)
        return_values.append(class_counts)

    return tuple(return_values)


@dataclass
class MultiInput:
    """
    A container for multiple input types with GPU support.

    This class is a dataclass that holds the fully connected and CNN input
    tensors, and provides methods for moving them to a device, detaching them
    from the computation graph, and getting their properties.
    """

    fc_input: torch.Tensor
    cnn_input: torch.Tensor

    def to(self, device: str) -> "MultiInput":
        """
        Moves all tensors to the specified device.

        Args:
            device (str): The device to move the tensors to.

        Returns:
            MultiInput: A new MultiInput object with the tensors on the
                specified device.
        """
        return MultiInput(
            fc_input=self.fc_input.to(device),
            cnn_input=self.cnn_input.to(device),
        )

    def cuda(self) -> "MultiInput":
        """
        Moves all tensors to the CUDA device.

        Returns:
            MultiInput: A new MultiInput object with the tensors on the CUDA
                device.
        """
        return self.to("cuda")

    def cpu(self) -> "MultiInput":
        """
        Moves all tensors to the CPU device.

        Returns:
            MultiInput: A new MultiInput object with the tensors on the CPU
                device.
        """
        return self.to("cpu")

    def detach(self) -> "MultiInput":
        """
        Detaches all tensors from the computation graph.

        Returns:
            MultiInput: A new MultiInput object with the detached tensors.
        """
        return MultiInput(
            fc_input=self.fc_input.detach(), cnn_input=self.cnn_input.detach()
        )

    @property
    def device(self) -> torch.device:
        """
        Gets the device of the tensors.

        Returns:
            torch.device: The device of the tensors.
        """
        return self.fc_input.device

    def __len__(self) -> int:
        """
        Returns the batch size.

        Returns:
            int: The batch size.
        """
        return len(self.fc_input)

    def __getitem__(self, idx: int) -> "MultiInput":
        """
        Gets a specific sample from the batch.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            MultiInput: A new MultiInput object containing the specified
                sample.
        """
        return MultiInput(
            fc_input=self.fc_input[idx], cnn_input=self.cnn_input[idx]
        )

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the `fc_input` tensor.

        Returns:
            torch.Size: The shape of the `fc_input` tensor.
        """
        return self.fc_input.shape


class CustomDataset(Dataset):
    """
    A custom dataset class for the project's data format.

    This class can return data as either a dictionary or a tuple of
    (inputs, target), which is useful for different models.

    Args:
        data_dict (dict): A dictionary containing the data.
        num_classes (int, optional): The number of classes in the dataset.
            If not provided, it will be inferred from the data. Defaults to
            None.
        return_tuple (bool, optional): Whether to return the data as a tuple
            of (inputs, target) or a dictionary. Defaults to False.
    """

    def __init__(
        self,
        data_dict: Dict[str, Any],
        num_classes: Optional[int] = None,
        return_tuple: bool = False,
    ):
        super(CustomDataset, self).__init__()
        self.data_dict = data_dict
        self.num_classes = num_classes
        self.return_tuple = return_tuple

        if self.num_classes is None:
            self.num_classes = (
                torch.max(self.data_dict["primary_array"]).item() + 1
            )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data_dict["xmax_array"])

    def __getitem__(self, idx: int) -> Any:
        """
        Gets a sample from the dataset.

        Args:
            idx (int): The index of the sample to get.

        Returns:
            Any: The sample, either as a dictionary or a tuple.
        """
        ID = self.data_dict["sd_event_id"][idx]
        xmax = self.data_dict["xmax_array"][idx]
        primary = self.data_dict["primary_array"][idx]
        shower_plane = self.data_dict["shower_plane_array"][idx]

        if self.return_tuple:
            return (xmax, shower_plane), primary
        else:
            sample = {
                "ID": ID,
                "fc_input": xmax,
                "cnn_input": shower_plane,
                "target": primary,
                "one_hot": F.one_hot(primary, num_classes=self.num_classes),
            }
            return sample