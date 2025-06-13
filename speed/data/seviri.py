
"""
speed.data.seviri
=================

This module contains functionality to add MSG SEVIRI observations to collocations.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import gc
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List
import warnings

import click
import numpy as np
from pansat import TimeRange, FileRecord
from pansat.time import to_datetime64
from pansat.utils import resample_data
from pansat.products.satellite.meteosat import (
    l1b_rs_msg_seviri
)
from pyresample.geometry import SwathDefinition
from rich.progress import track, Progress
from satpy import Scene
import xarray as xr

from speed.data.utils import round_time,
import speed.logging


LOGGER = logging.getLogger(__name__)


def add_seviri_obs(
        path_on_swath: Path,
        path_gridded: Path,
        n_steps: int = 4,
        sector: str = "conus"
) -> None:
    """
    Add SEVIRI observations for extracted collocations.

    Args:
        path_on_swath: Path to the file containing the collocations extracted in on_swath format.
        path_gridded: Path to the file containing the collocations extract in gridded format.
        n_steps: The number 10 minute time steps to extract.
        sector: A string specifying whether to load the data the full disk or only the CONUS
            sector.
    """
    time_step = np.timedelta64(10, "m")

    try:
        data = xr.open_dataset(path_on_swath, group="geo")
        data.close()
        LOGGER.info(
            "Skipping input files %s because they already contain geostationary observations.",
            path_on_swath
        )
        return None
    except Exception:
        pass

    LOGGER.info(
        "Processing %s.",
        path_on_swath
    )

    median_time = get_median_time(path_on_swath)
    rounded = to_datetime64(round_time(median_time, time_step))
    offsets = (np.arange(-n_steps // 2, n_steps // 2) + 1) * time_step
    time_steps = rounded + offsets

    seviri_recs = []
    for time_step in time_steps:
        time_range = TimeRange(time_step - np.datetime64(150, "s"), time_step + np.datetime64(149, "s"))
        seviri_rec = l1b_rs_msg_seviri.get(time_range)
        seviri_recs.append(seviri_rec[0])

    with xr.open_dataset(path_gridded, group="input_data") as data_gridded:
        lons_g = data_gridded.longitude.data
        lon_min_g, lon_max_g = lons_g.min(), lons_g.max()
        lats_g = data_gridded.latitude.data
        lat_min_g, lat_max_g = lats_g.min(), lats_g.max()
    del data_gridded

    with xr.open_dataset(path_on_swath, group="input_data") as data_on_swath:
        lons_n = data_on_swath.longitude.data
        lats_n = data_on_swath.latitude.data
        lat_min, lat_max = lats_g.min(), lats_g.max()
    del data_on_swath

    lons, lats = np.meshgrid(lons_g, lats_g)
    grid = SwathDefinition(xr.DataArray(lons), xr.DataArray(lats))
    swath = SwathDefinition(lons=xr.DataArray(lons_n), lats=xr.DataArray(lats_n))

    seviri_data_g = []
    seviri_data_s = []
    times = []

    for time, seviri_rec in zip(time_steps, seviri_recs):

        seviri_data = l1b_rs_msg_seviri.open(seviri_rec)

        hrv = data.HRV.coarsen({"latitude_0": 2 ,"longitude_0": 2}).mean()
        seviri_data = seviri_data.drop_vars("hrv")
        seviri_data["HRV"] = (("y_1", "x_1"), hrv.data)
        seviri_data = seviri_data.rename({
            "latitude_1": "latitude",
            "longitude_1": "longitude",
        })

        seviri_channels = [
            "HRV", "VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073", "IR_087",
            "IR_097", "IR_108", "IR_120", "IR_134"
        ]

        data_g = resample_data(seviri_data, grid)
        obs_g = np.stack([data_g[chan].data for chan in seviri_channels])
        seviri_data_g.append(obs_g)
        del data_g

        data_s = resample_data(seviri_data, swath)
        obs_s = np.stack([data_s[chan].data for chan in seviri_channels])
        seviri_data_s.append(obs_s)
        del data_s

        times.append(time)


    times = np.array(times)
    LOGGER.info(
        "Saving SEVIRI data for collocation %s.",
        time_str
    )

    seviri_data_g = xr.Dataset(
        {
            "latitude": (("latitude"), lats_g.astype(np.float32)),
            "longitude": (("longitude"), lons_g.astype(np.float32)),
            "time": (("time",), times),
            "observations": (
                ("latitude", "longitude", "time", "channel"),
                np.stack(seviri_data_g, 2).astype(np.float32)
            ),
        }
    )
    seviri_data_g.observations.encoding = {"dtype": "float32", "zlib": True}
    encoding = {
        var: {"zlib": True} for var in seviri_data_g
    }
    seviri_data_g.to_netcdf(path_gridded, group="geo", mode="a", encoding=encoding)

    seviri_data_s = xr.Dataset(
        {
            "observations": (
                ("scan", "pixel", "time", "channel"),
                np.stack(seviri_data_s, 2).astype(np.float32)
            ),
            "time": (("time",), times),
        }
    )
    seviri_data_s.observations.encoding = {"dtype": "float32", "zlib": True}
    encoding = {
        var: {"zlib": True} for var in seviri_data_s
    }
    seviri_data_s.to_netcdf(path_on_swath, group="geo", mode="a", encoding=encoding)

    del times
    del seviri_data_g
    del seviri_data_n
    gc.collect()



@click.command()
@click.argument("collocation_path", type=str)
@click.option("--n_steps", type=int, default=8)
@click.option("--n_processes", type=int, default=1)
@click.option("--pattern", type=str, default="*.nc")
def cli(
        collocation_path: str,
        n_steps: int = 8,
        n_processes: int = 1,
        pattern: str = "*.nc"
):
    """
    Extract SEVIRI observations matching GPM collocations.

    speed extract_goes collocation_path --n_steps N

    Extracts SEVIRI observations for all collocations found in 'collocation_path' in both gridded
    and on_swath projections. 'N' defines the number of 10-minute time steps centered on the
    median overpass time are extracted.
    """
    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error("Provided collocation path must point to an existing directory.")
        return 1

    files_on_swath = sorted(list((collocation_path / "on_swath").glob(pattern)))
    files_gridded = sorted(list((collocation_path / "gridded").glob(pattern)))

    times_on_swath = {}
    for f_on_swath in files_on_swath:
        time_str = f_on_swath.name.split("_")[2][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_on_swath[median_time] = f_on_swath

    times_gridded = {}
    for f_gridded in files_gridded:
        time_str = f_gridded.name.split("_")[2][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_gridded[median_time] = f_gridded

    combined = set(times_gridded.keys()).intersection(set(times_on_swath.keys()))
    combined = np.random.permutation(sorted(list(combined)))

    LOGGER.info(f"Found {len(combined)} collocations in {collocation_path}.")

    if n_processes < 2:
        for median_time in track(
                combined,
                description="Extracting SEVIRI observations:",
                console=speed.logging.get_console()
        ):
            try:
                add_seviri_obs(
                    times_on_swath[median_time],
                    times_gridded[median_time],
                    n_steps=n_steps
                )
            except Exception:
                LOGGER.exception(
                    "Processing of the collocation with median time %s failed "
                    "with the following error.",
                    median_time
                )
    else:
        pool = ProcessPoolExecutor(
            max_workers=n_processes
        )
        tasks = []
        for median_time in combined:
            tasks.append(pool.submit(
                add_seviri_obs,
                times_on_swath[median_time],
                times_gridded[median_time],
                n_steps=n_steps
            ))
        with Progress(console=speed.logging.get_console()) as progress:
            extraction = progress.add_task("Extracting SEVIRI observations:", total=len(tasks))
            for task in as_completed(tasks):
                try:
                    task.result()
                except Exception:
                    LOGGER.exception(
                        "The following error was encountered when processing collocation "
                        "with median time %s.",
                        median_time
                    )
                progress.advance(extraction, advance=1.0)
