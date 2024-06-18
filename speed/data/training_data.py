"""
speed.training_data
===================

Defines dataset classes to load SPEED training data.
"""
import os
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from .tiling import Tiler

class SPEED2D():
    def __init__(
            self,
            path,
            augment: bool = True,
            rqi_threshold: float = 0.8
    ):

        self.files = np.array(sorted(list(Path(path).glob("*.nc"))))
        self.augment = augment
        self.rqi_threshold = rqi_threshold
        self.init_rng()


    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def worker_init_fn(self, *args):
        """
        Pytorch retrieve interface.
        """
        return self.init_rng(*args)


    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, index):

        with xr.open_dataset(self.files[index]) as input_data:

            x = np.transpose(input_data.tbs_mw.data, (2, 0, 1)).astype("float32")
            precip = input_data.surface_precip.data.astype("float32")

            if self.rqi_threshold is not None:
                rqi = input_data.radar_quality_index.data
                invalid = rqi < self.rqi_threshold
            precip[invalid] = np.nan
            y = precip

            if self.augment:
                flip_h = self.rng.random() > 0.5
                if flip_h:
                    x = np.flip(x, -1)
                    y = np.flip(y, -1)
                flip_v = self.rng.random() > 0.5
                if flip_v:
                    x = np.flip(x, -2)
                    y = np.flip(y, -2)
                transp = self.rng.random() > 0.5
                if transp:
                    x = np.transpose(x, (0, 2, 1))
                    y = np.transpose(y, (1, 0))

        x = torch.tensor(x.copy())
        y = torch.tensor(y.copy())

        return x, y


class CollocationLoader:
    """
    Loads retrieval input data from the collocation files.
    """

    def __init__(self, path: Path, input_size, overlap=32):
        self.path = path

        with xr.open_dataset(path, group="input_data") as input_data:
            tbs = np.transpose(input_data.tbs_mw.data, (2, 0, 1))
        self.x = torch.tensor(tbs.astype("float32"))
        self.tiler = Tiler(self.x, tile_size=input_size, overlap=overlap)

    def __iter__(self) -> torch.Tensor:
        """
        Iterates over tiles and assembles results.
        """
        yield from self.tiler
        #gen = iter(self.tiler)
        #for tile in gen:
        #    result = yield tile
        #    print(result)
        #    gen.send(result)
        #return tile
