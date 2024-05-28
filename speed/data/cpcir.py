"""
speed.data.cpcir
================

This module contains functionality to add CPCIR (MERGEDIR) observations to collocations.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import logging
from pathlib import Path

import click

import numpy as np
from pansat import TimeRange
from pansat.time import to_datetime64
from pansat.products.satellite.gpm import merged_ir
from pyresample.geometry import SwathDefinition
from rich.progress import Progress
import xarray as xr

from speed.data.utils import resample_data, round_time


LOGGER = logging.getLogger(__name__)


def add_cpcir_obs(
        path_native: Path,
        path_gridded: Path,
        n_steps = 8
):
    """
    Add CPCIR data for extracted collocations.

    Args:
        path_native: Path to the file containing the collocations extracted in native format.
        path_gridded: Path to the file containing the collocations extract in gridded format.
        n_steps: The number 30 minute time steps to extract.
    """
    time_str = path_native.name.split("_")[2][:-3]
    median_time = to_datetime64(datetime.strptime(time_str, "%Y%m%d%H%M%S"))
    rounded = round_time(median_time, np.timedelta64(30, "m"))
    offsets = (np.arange(-n_steps // 2, n_steps // 2) + 1) * np.timedelta64("30", "m")
    time_steps = rounded - offsets
    cpcir_recs = merged_ir.get(TimeRange(time_steps.min(), time_steps.max()))

    with xr.open_dataset(path_gridded, group="input_data") as data_gridded:
        lons_g = data_gridded.longitude.data
        lon_min_g, lon_max_g = lons_g.min(), lons_g.max()
        lats_g = data_gridded.latitude.data
        lat_min_g, lat_max_g = lats_g.min(), lats_g.max()

    with xr.open_dataset(path_native, group="input_data") as data_native:
        lons_n = data_native.longitude.data
        lats_n = data_native.latitude.data
        lon_min_n, lon_max_n = lons_n.min(), lons_n.max()
        lat_min_n, lat_max_n = lats_n.min(), lats_n.max()

    swath = SwathDefinition(lons=lons_n, lats=lats_n)

    cpcir_data_g = []
    cpcir_data_n = []

    for cpcir_rec in cpcir_recs:
        with xr.open_dataset(cpcir_rec.local_path) as data_t:

            data_t = data_t.rename({"lat": "latitude", "lon": "longitude"})

            lons = data_t.longitude.data
            lon_inds = np.where((lons >= lon_min_g) * (lons <= lon_max_g))[0]
            lats = data_t.latitude.data
            lat_inds = np.where((lats >= lat_min_g) * (lats <= lat_max_g))[0]
            data_g = data_t[{"latitude": lat_inds, "longitude": lon_inds}]
            data_g = data_g.interpolate_na(dim="longitude")
            cpcir_data_g.append(data_g.interp(latitude=lats_g, longitude=lons_g, method="nearest"))

            lon_inds = np.where((lons >= lon_min_n) * (lons <= lon_max_n))[0]
            lat_inds = np.where((lats >= lat_min_n) * (lats <= lat_max_n))[0]
            data_n = data_t[{"latitude": lat_inds, "longitude": lon_inds}]
            data_n = data_n.interpolate_na(dim="longitude")
            lons = xr.DataArray(data=lons_n, dims=(("scans", "pixels")))
            lats = xr.DataArray(data=lats_n, dims=(("scans", "pixels")))
            cpcir_data_n.append(data_n.interp(latitude=lats, longitude=lons, method="nearest"))


    # Save data in gridded format.
    cpcir_data_g = xr.concat(cpcir_data_g, "time").sortby("time").rename({"Tb": "tbs_ir"})
    cpcir_data_g.tbs_ir.encoding = {"dtype": "uint16", "_FillValue": 0, "scale_factor": 0.01}
    cpcir_data_g = cpcir_data_g.interp(time=time_steps.astype("datetime64[ns]"), method="nearest")
    input_data = xr.load_dataset(path_gridded, group="input_data")
    reference_data = xr.load_dataset(path_gridded, group="reference_data")
    input_data.to_netcdf(path_gridded, group="input_data")
    reference_data.to_netcdf(path_gridded, group="reference_data", mode="a")
    cpcir_data_g.to_netcdf(path_gridded, group="geo_ir", mode="a")

    # Save data in gridded format.
    cpcir_data_n = xr.concat(cpcir_data_n, "time").sortby("time").rename({"Tb": "tbs_ir"})
    cpcir_data_n.tbs_ir.encoding = {"dtype": "uint16", "_FillValue": 0, "scale_factor": 0.01}
    cpcir_data_n = cpcir_data_n.interp(time=time_steps.astype("datetime64[ns]"), method="nearest")
    input_data = xr.load_dataset(path_native, group="input_data")
    reference_data = xr.load_dataset(path_native, group="reference_data", mode="a")
    input_data.to_netcdf(path_native, group="input_data")
    reference_data.to_netcdf(path_native, group="reference_data", mode="a")
    cpcir_data_n.to_netcdf(path_native, group="geo_ir", mode="a")


@click.command()
@click.argument("collocation_path", type=str)
@click.option(
    "--time_range",
    type=int,
    default=8,
    help="The width of the time interval in hours for which to extract the CPCIR observations"
)
@click.option(
    "--n_processes",
    type=int,
    default=8,
    help="The number of processes to use for the data extraction."
)
def cli(
        collocation_path: str,
        time_range: int = 8,
        n_processes: int = 4
):
    """
    Extract CPCIR (MERGEDIR) observations for GPM collocations in COLLOCATION_PATH.

    Extracts CPCIR observations for all collocations found in 'collocation_path' in both gridded
    and native projections. 'time_range' defines the a time interval in hours  centered on the
    median overpass time for which observations are extracted.
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

    pool = ProcessPoolExecutor(max_workers=n_processes)
    tasks = []

    try:
        for median_time in combined:
            tasks.append(pool.submit(
                add_cpcir_obs,
                times_native[median_time],
                times_gridded[median_time],
                n_steps=time_range * 2
            ))

        with Progress() as progress:
            prog = progress.add_task("Extracting CPCIR observations:", total=len(tasks))
            for median_time, task in zip(combined, tasks):
                try:
                    task.result()
                except KeyboardInterrupt as exc:
                    raise exc
                except Exception as exc:
                    LOGGER.exception(
                        "Encountered an error when processing collocation with median overpass time %s.",
                        median_time
                    )

                progress.update(prog, advance=1)

    except KeyboardInterrupt:
        for task in tasks:
            task.cancel()
        pool.shutdown(wait=False)
