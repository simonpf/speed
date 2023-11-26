"""
spree.data.reference
====================

This module defines the representation of reference data products.
"""
from abc import ABC, abstractmethod
from typing import List

from pansat import FileRecord, TimeRange
from pansat.products.ground_based import mrms


ALL_DATASETS = {}


class ReferenceData(ABC):
    """
    Class representing reference precipitation measurements.
    """

    def __init__(self, name, domain, pansat_product):
        self.name = name
        self.domain = domain
        self.pansat_product = pansat_product
        ALL_DATASETS[self.name] = self

    @abstractmethod
    def load_reference_data(self, time_range: TimeRange):
        """
        Load, resample and combine reference data with IR obs for
        a given time range.

        Args:
            time_range: A time range object defining a time range for
                which to load reference data.

        Return:
            A xarray.Dataset containing regridded reference data
            combined with IR brightness temperatures.
        """


def get_reference_dataset(name: str) -> ReferenceData:
    """
    Retrieve reference dataset by its name.

    Args:
        name: The name of the dataset.

    The dataset instance or 'None' if no such dataset is known.
    """
    return ALL_DATASETS.get(name, None)
