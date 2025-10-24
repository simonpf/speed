"""
spree.data.reference
====================

This module defines the representation of reference data products.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from pansat.granule import Granule


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
    def load_reference_data(
            self,
            inpt_granule: Granule,
            ref_granules: List[Granule],
            radius_of_influence: float,
            beam_width: Optional[float]
    ):
        """
        Load and resample reference data to regular lat/lon grid.

        Args:
            inpt_granule: The input-data granule to which the reference
                data is to be mapped.
            ref_granules: A list of reference granules matching the input
                input granule.
            radius_of_influence: The approximate spatial resolution of
                each input-data pixel.
            beam_width: The beam width to assume for the calculation
                of footprint-averaged data.

        Return:
            A tuple ``(ref_data, ref_data_fpavg)`` containing the gridded
            reference data in ``ref_data`` and the footprint-averaged reference
            data in ``ref_data_fpavg``.
        """


def get_reference_dataset(name: str) -> ReferenceData:
    """
    Retrieve reference dataset by its name.

    Args:
        name: The name of the dataset.

    The dataset instance or 'None' if no such dataset is known.
    """
    return ALL_DATASETS.get(name, None)
