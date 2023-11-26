"""
speed.data.input
================

Defines the base class for all types of input data.
"""
ALL_DATASETS = {}


class InputData:
    """
    Base class for all input datasets. Keeps track of all initiations
    and allows accessing the by name.
    """

    def __init__(self, name):
        self.name = name
        ALL_DATASETS[name] = self


def get_input_dataset(name: str) -> InputData:
    """
    Retrieve input dataset by its name.

    Args:
        name: The name of the dataset.

    The dataset instance or 'None' if no such dataset is known.
    """
    return ALL_DATASETS.get(name, None)
