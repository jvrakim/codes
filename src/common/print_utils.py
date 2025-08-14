"""
Provides utility functions for printing formatted output to the console.

This module contains functions for printing message boxes and tables, as well
as a class for displaying progress bars using the `rich` library.
"""

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.table import Table
from rich.console import Group, Console
from typing import List, Optional


def print_msg_box(
    msg: str,
    indent: int = 1,
    width: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Prints a message in a box.

    Args:
        msg (str): The message to print.
        indent (int, optional): The indentation of the message. Defaults to 1.
        width (int, optional): The width of the box. If not provided, it will
            be calculated automatically. Defaults to None.
        title (str, optional): The title of the box. Defaults to None.
    """
    lines = msg.split("\n")
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f"╔{'═' * (width + indent * 2)}╗\n"
    if title:
        box += f"║{space}{title:<{width}}{space}║\n"
        box += f"║{space}{'-' * len(title):<{width}}{space}║\n"
    box += "".join([f"║{space}{line:<{width}}{space}║\n" for line in lines])
    box += f"╚{'═' * (width + indent * 2)}╝"
    print(box)


class RichProgress:
    """
    A class for displaying progress bars and metrics using the `rich` library.

    Args:
        num_batches (int): The total number of batches.
        model_names (list of str): A list of model names to display metrics
            for.
    """

    def __init__(self, num_batches: int, model_names: List[str]):
        self.console = Console()
        self.model_names = model_names
        self.progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total} {task.fields[unit]}"),
            transient=True,
        )
        self.task_id = self.progress.add_task(
            "train", total=num_batches, unit="batch"
        )
        self.metrics = {name: {"loss": "", "acc": ""} for name in model_names}

    def set_description(self, desc: str):
        """
        Sets the description of the progress bar.

        Args:
            desc (str): The new description.
        """
        self.progress.update(self.task_id, description=desc)

    def update(self, batch_size: int = 1):
        """
        Updates the progress bar.

        Args:
            batch_size (int, optional): The number of batches to advance the
                progress bar by. Defaults to 1.
        """
        self.progress.update(self.task_id, advance=batch_size)

    def set_metrics(
        self, names: List[str], loss_strs: List[str], acc_strs: List[str]
    ):
        """
        Sets the metrics to display.

        Args:
            names (list of str): A list of model names.
            loss_strs (list of str): A list of loss strings.
            acc_strs (list of str): A list of accuracy strings.
        """
        if not (len(names) == len(loss_strs) == len(acc_strs)):
            raise ValueError(
                "names, loss_strs, and acc_strs must all have the same length"
            )
        for name, lstr, astr in zip(names, loss_strs, acc_strs):
            if name not in self.metrics:
                raise KeyError(f"Unknown model name: {name}")
            self.metrics[name]["loss"] = lstr
            self.metrics[name]["acc"] = astr

    def __enter__(self):
        """Enters the context manager."""
        self.group = Group(self.progress, self._render_table())
        self.live = Live(
            self.group,
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exits the context manager."""
        self.live.__exit__(exc_type, exc, tb)

    def _render_table(self) -> Table:
        """
        Renders the metrics table.

        Returns:
            Table: The rendered table.
        """
        table = Table(expand=True)
        table.add_column("Model", justify="left")
        table.add_column("Loss", justify="right")
        table.add_column("Accuracy", justify="right")
        for name in self.model_names:
            table.add_row(
                name, self.metrics[name]["loss"], self.metrics[name]["acc"]
            )
        return table

    def refresh(self):
        """Refreshes the display."""
        self.live.update(Group(self.progress, self._render_table()))


def get_model_info_table(model_info: dict) -> Table:
    """
    Creates a `rich` table to display model information.

    Args:
        model_info (dict): A dictionary containing model information.
            Expected keys: 'total_parameters', 'trainable_parameters',
            'model_size_mb'.

    Returns:
        Table: A `rich` table object.
    """
    table = Table(title="Model Information")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="magenta")

    table.add_row(
        "Total Parameters", f"{model_info['total_parameters']:,}"
    )
    table.add_row(
        "Trainable Parameters", f"{model_info['trainable_parameters']:,}"
    )
    table.add_row("Model Size (MB)", f"{model_info['model_size_mb']:.2f}")

    return table
