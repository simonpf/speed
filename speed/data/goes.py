"""
speed.data.goes
===============

This module contains functionality to add GOES 16 observations to collocations.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List
import warnings

import click
import numpy as np
from pansat import TimeRange, FileRecord
from pansat.time import to_datetime64
from pansat.products.satellite.goes import (
    GOES16L1BRadiances
)
from pyresample.geometry import SwathDefinition
from rich.progress import track, Progress
from satpy import Scene
import xarray as xr

from speed.data.utils import round_time
import speed.logging


LOGGER = logging.getLogger(__name__)



def find_goes_files(
        products,
        start_time: np.datetime64,
        end_time: np.datetime64,
        time_step: np.timedelta64,
) -> Dict[np.datetime64, List[FileRecord]]:
    """
    Find input data files within a given file range from which to extract
    training data.

    Args:
        start_time: Start time of the time range.
        end_time: End time of the time range.
        time_step: The time step of the retrieval.
        roi: An optional geometry object describing the region of interest
        that can be used to restriced the file selection a priori.
        path: If provided, files should be restricted to those available from
        the given path.

    Return:
        A list of list of files to extract the GOES retrieval input data from.
    """
    found_files = {}

    time_range = TimeRange(start_time, end_time).expand(np.timedelta64(8 * 60, "s"))

    for prod in products:
        recs = prod.find_files(time_range)

        matched_recs = {}
        matched_deltas = {}

        for rec in recs:
            tr_rec = rec.temporal_coverage
            time_c = to_datetime64(tr_rec.start + 0.5 * (tr_rec.end - tr_rec.start))
            time_n = round_time(time_c, time_step)
            delta = abs(time_c - time_n)

            min_delta = matched_deltas.get(time_n)
            if min_delta is None:
                matched_recs[time_n] = rec
                matched_deltas[time_n] = delta
            else:
                if delta < min_delta:
                    matched_recs[time_n] = rec
                    matched_deltas[time_n] = delta

        for time_n, matched_rec in matched_recs.items():
            found_files.setdefault(time_n, []).append(matched_rec)

    return found_files


def add_goes_obs(
        path_native: Path,
        path_gridded: Path,
        n_steps: int = 4,
        sector: str = "full"
):
    """
    Add GOES observations for extracted collocations.

    Args:
        path_native: Path to the file containing the collocations extracted in native format.
        path_gridded: Path to the file containing the collocations extract in gridded format.
        n_steps: The number 30 minute time steps to extract.
        sector: A string specifying whether to load the data the full disk or only the CONUS
            sector.
    """
    time_step = np.timedelta64(15, "m")

    time_str = path_native.name.split("_")[2][:-3]
    median_time = to_datetime64(datetime.strptime(time_str, "%Y%m%d%H%M%S"))
    rounded = round_time(median_time, time_step)
    offsets = (np.arange(-n_steps // 2, n_steps // 2) + 1) * time_step
    time_steps = rounded - offsets

    if sector.lower() == "conus":
        products = [GOES16L1BRadiances("C", channel) for channel in range(1, 17)]
    else:
        products = [GOES16L1BRadiances("F", channel) for channel in range(1, 17)]


    goes_recs = find_goes_files(products, time_steps.min(), time_steps.max(), time_step)
    assert all([step in goes_recs for step in time_steps])

    with xr.open_dataset(path_gridded, group="input_data") as data_gridded:
        lons_g = data_gridded.longitude.data
        lon_min_g, lon_max_g = lons_g.min(), lons_g.max()
        lats_g = data_gridded.latitude.data
        lat_min_g, lat_max_g = lats_g.min(), lats_g.max()

    with xr.open_dataset(path_native, group="input_data") as data_native:
        lons_n = data_native.longitude.data
        lats_n = data_native.latitude.data
        lat_min, lat_max = lats_g.min(), lats_g.max()

    lons, lats = np.meshgrid(lons_g, lats_g)
    grid = SwathDefinition(xr.DataArray(lons), xr.DataArray(lats))
    swath = SwathDefinition(lons=xr.DataArray(lons_n), lats=xr.DataArray(lats_n))

    goes_data_g = []
    goes_data_n = []

    with TemporaryDirectory() as tmp:
        for time in time_steps:
            try:
                logging.disable(logging.CRITICAL)
                recs = [rec.get() for rec in goes_recs[time]]

                with warnings.catch_warnings():
                    scene = Scene([str(rec.local_path) for rec in recs])
                    scene.load([f"C{ind:02}" for ind in range(1, 17)])
                    scene_g = scene.resample(grid, generate=False, cache_dir=tmp)
                goes_obs = scene_g.to_xarray_dataset()
                goes_obs = np.stack([goes_obs[f"C{ind:02}"] for ind in range(1, 17)], -1)
                goes_data_g.append(goes_obs)

                with warnings.catch_warnings():
                    scene_n = scene.resample(swath, generate=False, cache_dir=tmp)
                goes_obs = scene_n.to_xarray_dataset()
                goes_obs = np.stack([goes_obs[f"C{ind:02}"] for ind in range(1, 17)], -1)
                goes_data_n.append(goes_obs)

                del scene
                del scene_g
                del scene_n
            finally:
                logging.disable(logging.NOTSET)

        # Save data in gridded format.

    goes_data_g = xr.Dataset(
        {
            "latitude": (("latitude"), lats_g.astype(np.float32)),
            "longitude": (("longitude"), lons_g.astype(np.float32)),
            "observations": (
                ("latitude", "longitude", "time", "channel"),
                np.stack(goes_data_g, 2).astype(np.float32)
            ),
        },
        encoding = {
            "observations": {"dtype": "float32", "zlib": True}
        }
    )

    LOGGER.info(
        "Saving GOES data for collocation %s.",
        time_str
    )

    goes_data_g.to_netcdf(path_gridded, group="geo")
    goes_data_n.to_netcdf(path_native, group="geo")



@click.command()
@click.argument("collocation_path", type=str)
@click.option("--n_steps", type=int, default=8)
@click.option("--n_processes", type=int, default=1)
def cli(
        collocation_path: str,
        n_steps: int = 8,
        n_processes: int = 1
):
    """
    Extract GOES observations matching GPM collocations.

    speed extract_goes collocation_path --n_steps N

    Extracts GOES-16 observations for all collocations found in 'collocation_path' in both gridded
    and native projections. 'N' defines the number of half-hourly time steps centered on the
    median overpass time are extracted.
    """
    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error("Provided collocation path must point to an existing directory.")
        return 1

    files_native = sorted(list((collocation_path / "native").glob("*.nc")))
    files_gridded = sorted(list((collocation_path / "gridded").glob("*.nc")))

    times_native = {}
    for f_native in files_native:
        time_str = f_native.name.split("_")[2][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_native[median_time] = f_native

    times_gridded = {}
    for f_gridded in files_gridded:
        time_str = f_gridded.name.split("_")[2][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_gridded[median_time] = f_gridded

    combined = set(times_gridded.keys()).intersection(set(times_native.keys()))

    LOGGER.info(f"Found {len(combined)} collocations in {collocation_path}.")

    if n_processes < 2:
        for median_time in track(
                combined,
                description="Extracting GOES observations:",
                console=speed.logging.get_console()
        ):
            try:
                add_goes_obs(
                    times_native[median_time],
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
                add_goes_obs,
                times_native[median_time],
                times_gridded[median_time],
                n_steps=n_steps
            ))
        with Progress(console=speed.logging.get_console()) as progress:
            task = progress.add_task("Extracting GOES observations:", total=len(tasks))
            for task in as_completed(tasks):
                try:
                    task.result()
                except Exception:
                    LOGGER.exception(
                        "The following error was encountered when processing collocation "
                        "with median time %s.",
                        median_time
                    )
                progress.advance(task, advance=1.0)
