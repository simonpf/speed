"""
speed.grids
===========

Define the global grid used by speed and provides helper functions
to access its attributes.
"""
from dataclasses import dataclass
import numpy as np
from pathlib import Path

import pyresample


@dataclass
class Grid:
    """
    Wrapper class combining relevant aspects of the grid used by
    SPEED.
    """

    grid: pyresample.geometry.AreaDefinition
    lons: np.ndarray
    lats: np.ndarray

    def __init__(self, area):
        self.grid = area
        lons, lats = area.get_lonlats()
        self.lons = lons[0]
        self.lats = lats[..., 0]


GLOBAL = Grid(pyresample.load_area(Path(__file__).parent / "global.yml"))
