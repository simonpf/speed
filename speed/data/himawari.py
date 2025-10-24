"""
speed.data.himawari
===================

This module contains functionality to add HIMAWARI 9 observations to collocations.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path

import click
import numpy as np
from pansat.time import to_datetime64
from pansat.utils import resample_data
from pansat.products.satellite.himawari import (
    l1b_himawari8_all,
    l1b_himawari9_all
)
from pyresample.geometry import SwathDefinition
from rich.progress import track, Progress
from satpy import Scene
import xarray as xr

from speed.data.utils import round_time
import speed.logging


LOGGER = logging.getLogger(__name__)


def add_himawari_obs(
        path_on_swath: Path,
        path_gridded: Path,
        n_steps: int = 4,
) -> None:
    """
    Add HIMAWARI observations for extracted collocations.

    Args:
        path_on_swath: Path to the file containing the collocations extracted in on_swath format.
        path_gridded: Path to the file containing the collocations extract in gridded format.
        n_steps: The number 30 minute time steps to extract.
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
    except (KeyError, ValueError):
        # No geo group exists yet, proceed with processing
        LOGGER.debug("No existing geo data found in %s, proceeding with processing", path_on_swath)
    except Exception as e:
        LOGGER.warning("Error checking for existing geo data in %s: %s", path_on_swath, e)
        # Continue processing despite check failure

    time_str = path_on_swath.name.split("_")[2][:-3]
    median_time = to_datetime64(datetime.strptime(time_str, "%Y%m%d%H%M%S"))
    rounded = round_time(median_time, time_step)
    offsets = (np.arange(-n_steps // 2, n_steps // 2) + 1) * time_step
    time_steps = rounded + offsets

    if median_time < np.datetime64("2022-12-01"):
        prod = l1b_himawari8_all
    else:
        prod = l1b_himawari9_all

    himawari_recs = {}
    for time in time_steps:
        himawari_recs[time] = prod.find_files(time)

    assert all([step in himawari_recs for step in time_steps])

    with xr.open_dataset(path_gridded, group="input_data") as data_gridded:
        lons_g = data_gridded.longitude.data
        _lon_min_g, _lon_max_g = lons_g.min(), lons_g.max()
        lats_g = data_gridded.latitude.data
        _lat_min_g, _lat_max_g = lats_g.min(), lats_g.max()
    del data_gridded

    with xr.open_dataset(path_on_swath, group="input_data") as data_on_swath:
        lons_n = data_on_swath.longitude.data
        lon_min, lon_max = lons_n.min(), lons_n.max()
        lats_n = data_on_swath.latitude.data
        lat_min, lat_max = lats_n.min(), lats_n.max()
    del data_on_swath

    lons, lats = np.meshgrid(lons_g, lats_g)
    grid = SwathDefinition(xr.DataArray(lons), xr.DataArray(lats))
    swath = SwathDefinition(lons=xr.DataArray(lons_n), lats=xr.DataArray(lats_n))

    himawari_data_g = []
    himawari_data_n = []
    times = []

    for time in time_steps:
        try:
            obs_g = []
            obs_n = []
            recs = [rec.get() for rec in himawari_recs[time]]
            for band in range(1, 17):
                band_recs = [rec for rec in recs if f"B{band:02}" in rec.filename]
                logging.disable(logging.CRITICAL)

                scene = Scene([str(rec.local_path) for rec in band_recs])
                scene.load([f"B{band:02}"])
                area = scene.coarsest_area()
                lons, lats = area.get_lonlats()

                lat_mask = ((lat_min <= lats) * (lats <= lat_max)).any(1)
                lat_start = np.where(lat_mask)[0].min()
                lat_end = np.where(lat_mask)[0].max()
                lon_mask = ((lon_min <= lons) * (lons <= lon_max)).any(0)
                lon_start = np.where(lon_mask)[0].min()
                lon_end = np.where(lon_mask)[0].max()

                scene_r = scene[lat_start:lat_end, lon_start:lon_end]
                data = scene_r.to_xarray_dataset().compute()
                data["longitude"] = (("y", "x"), lons[lat_start:lat_end, lon_start:lon_end])
                data["latitude"] = (("y", "x"), lats[lat_start:lat_end, lon_start:lon_end])
                data_g = resample_data(data, grid, radius_of_influence=5e3)
                data_n = resample_data(data, swath, radius_of_influence=5e3)
                channel = band
                name = f"B{channel:02}"
                obs_g.append(xr.Dataset({
                    "channel": channel,
                    "observations": (("latitude", "longitude"), data_g[name].data)
                }))
                obs_n.append(xr.Dataset({
                    "channel": channel,
                    "observations": (("latitude", "longitude"), data_n[name].data)
                }))

                del data
                del data_g
                del data_n

            obs_g = xr.concat(obs_g, dim="channel").sortby("channel").transpose("latitude", "longitude", "channel")
            obs_n = xr.concat(obs_n, dim="channel").sortby("channel").transpose("latitude", "longitude", "channel")

            himawari_data_g.append(obs_g.observations.data)
            himawari_data_n.append(obs_n.observations.data)
            times.append(to_datetime64(recs[0].central_time.start))

        finally:
            logging.disable(logging.NOTSET)

        # Save data in gridded format.

    times = np.array(times)
    LOGGER.info(
        "Saving HIMAWARI data for collocation %s.",
        time_str
    )

    himawari_data_g = xr.Dataset(
        {
            "latitude": (("latitude"), lats_g.astype(np.float32)),
            "longitude": (("longitude"), lons_g.astype(np.float32)),
            "time": (("time",), times),
            "observations": (
                ("latitude", "longitude", "time", "channel"),
                np.stack(himawari_data_g, 2).astype(np.float32)
            ),
        }
    )
    himawari_data_g.observations.encoding = {"dtype": "float32", "zlib": True}
    himawari_data_g.to_netcdf(path_gridded, group="geo", mode="a")

    himawari_data_n = xr.Dataset(
        {
            "observations": (
                ("scan", "pixel", "time", "channel"),
                np.stack(himawari_data_n, 2).astype(np.float32)
            ),
            "time": (("time",), times),
        }
    )
    himawari_data_n.observations.encoding = {"dtype": "float32", "zlib": True}
    himawari_data_n.to_netcdf(path_on_swath, group="geo", mode="a")

    del himawari_data_g
    del himawari_data_n



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
    Extract HIMAWARI observations matching GPM collocations.

    speed extract_himawari collocation_path --n_steps N

    Extracts HIMAWARI observations for all collocations found in 'collocation_path' in both gridded
    and on_swath projections. 'N' defines the number of half-hourly time steps centered on the
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

    LOGGER.info(f"Found {len(combined)} collocations in {collocation_path}.")
    combined = sorted(list(combined))[:1]

    if n_processes < 2:
        for median_time in track(
                combined,
                description="Extracting HIMAWARI observations:",
                console=speed.logging.get_console()
        ):
            try:
                add_himawari_obs(
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
                add_himawari_obs,
                times_on_swath[median_time],
                times_gridded[median_time],
                n_steps=n_steps
            ))
        with Progress(console=speed.logging.get_console()) as progress:
            extraction = progress.add_task("Extracting HIMAWARI observations:", total=len(tasks))
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
