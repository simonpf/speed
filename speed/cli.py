"""
speed.cli
=========

The command line interface of SPEED.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
from rich.progress import track
from typing import List
import xarray as xr

import click

import speed.logging
from speed.data import cpcir, goes

LOGGER = logging.getLogger(__file__)


@click.group()
def cli():
    pass


@click.command()
@click.argument("input_data")
@click.argument("reference_data")
@click.argument("output_folder")
@click.argument(
    "year",
    type=int,
)
@click.argument(
    "month",
    type=int,
)
@click.argument("days", type=int, nargs=-1)
@click.option('-n', '--n_processes', default=1)
def extract_data(
        input_data: str,
        reference_data: str,
        output_folder: str,
        year: int,
        month: int,
        days: List[int],
        n_processes: int = 1
):
    """
    Extract collocations for a given date.
    """
    from speed.data.input import get_input_dataset
    from speed.data.reference import get_reference_dataset
    import speed.data.gpm
    import speed.data.mrms
    import speed.data.gpm_gv
    import speed.data.combined

    input_dataset = get_input_dataset(input_data)
    if input_dataset is None:
        LOGGER.error(f"The input dataset '{input_data}' is not known.")
        return 1
    reference_dataset = get_reference_dataset(reference_data)
    if reference_dataset is None:
        LOGGER.error(f"The input dataset '{reference_data}' is not known.")
        return 1
    output_folder = Path(output_folder)

    if days is None or len(days) == 0:
        days = list(range(1, monthrange(year, month)[1] + 1))

    if n_processes < 2:
        for day in days:
            try:
                input_dataset.process_day(year, month, day, reference_dataset, output_folder)
            except Exception as exc:
                LOGGER.exception(exc)
    else:
        pool = ProcessPoolExecutor(max_workers=n_processes)
        manager = multiprocessing.Manager()

        tasks = {}
        for day in days:
            task = pool.submit(
                input_dataset.process_day,
                year,
                month,
                day,
                reference_dataset,
                output_folder,
            )
            tasks[task] = (year, month, day)

        for task in as_completed(tasks):
            try:
                task.result()
            except Exception as exc:
                LOGGER.exception(exc)




@click.command()
@click.argument("collocation_path")
@click.argument("output_folder")
@click.option(
    "--overlap",
    type=float,
    default=0.0,
)
@click.option(
    "--size",
    type=int,
    default=256,
)
@click.option("--include_geo", help="Include GEO observations.", is_flag=True, default=False)
@click.option("--include_geo_ir", help="Include geo IR observations.", is_flag=True, default=False)
def extract_training_data_spatial(
        collocation_path: str,
        output_folder: str,
        overlap: float = 0.0,
        size: int = 256,
        min_input_frac: float = None,
        include_geo: bool = False,
        include_geo_ir: bool = False
) -> int:
    """
    Extract spatial training scenes from collocations in COLLOCATION_PATH and write scenes
    to OUTPUT_FOLDER.
    """
    from speed.data.utils import extract_scenes

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error(
            "'collocation_path' must point to an existing directory."
        )
        return 1


    collocation_files = sorted(list(collocation_path.glob("*.nc")))
    for collocation_file in track(collocation_files, "Extracting spatial training data:"):

        sensor_name = collocation_file.name.split("_")[1]

        try:
            extract_scenes(
                collocation_file,
                output_folder,
                overlap=overlap,
                size=size,
                include_geo=include_geo,
                include_geo_ir=include_geo_ir,
            )
        except Exception:
            LOGGER.exception(
                "Encountered an error when processing file %s.",
                collocation_file
            )

    return 1


@click.command()
@click.argument("collocation_path")
@click.argument("output_folder")
@click.option("--include_geo", help="Include geo observations.", is_flag=True, default=False)
@click.option("--include_geo_ir", help="Include geo IR observations.", is_flag=True, default=False)
def extract_training_data_tabular(
        collocation_path: str,
        output_folder: str,
        include_geo: bool = False,
        include_geo_ir: bool = False
) -> int:
    """
    Extract tabular training data from collocations in COLLOCATION_PATH and write resulting files
    to OUTPUT_FOLDER.
    """
    from speed.data.utils import extract_training_data

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error(
            "'collocation_path' must point to an existing directory."
        )
        return 1


    collocation_files = sorted(list(collocation_path.glob("*.nc")))

    pmw_data = []
    geo_data = []
    geo_ir_data = []
    anc_data = []
    target_data = []

    for collocation_file in track(collocation_files, description="Extracting tabular training data:"):
        try:
            pmw, geo, geo_ir, anc, target, = extract_training_data(
                collocation_file,
                include_geo=include_geo,
                include_geo_ir=include_geo_ir
            )
            pmw_data.append(pmw)
            if geo is not None:
                geo_data.append(geo)
            if geo_ir is not None:
                geo_ir_data.append(geo_ir)
            anc_data.append(anc)
            target_data.append(target)

        except Exception:
            LOGGER.exception(
                "Encountered an error when processing file %s.",
                collocation_file
            )

    pmw_data = xr.concat(pmw_data, dim="samples")
    ancillary_data = xr.concat(anc_data, dim="samples")
    target_data = xr.concat(target_data, dim="samples")

    encoding = {
        "observations": {
            "zlib": True,
            "scale_factor": 0.01,
            "dtype": "uint16",
            "_FillValue": 2e16 - 1
        }
    }
    (output_folder / "pmw").mkdir(exist_ok=True)
    pmw_data.to_netcdf(
        output_folder / "pmw" / "pmw.nc",
    )


    (output_folder / "ancillary").mkdir(exist_ok=True)
    ancillary_data.to_netcdf(
        output_folder / "ancillary" / "ancillary.nc",
    )
    (output_folder / "target").mkdir(exist_ok=True)
    target_data.to_netcdf(
        output_folder / "target" / "target.nc",
    )
    if len(geo_data) > 0:
        geo_data = xr.concat(geo_data, dim="samples")
        (output_folder / "geo").mkdir(exist_ok=True)
        geo_data.to_netcdf(
            output_folder / "geo" / "geo.nc",
            encoding=encoding
        )
    if len(geo_ir_data) > 0:
        geo_ir_data = xr.concat(geo_ir_data, dim="samples")
        (output_folder / "geo_ir").mkdir(exist_ok=True)
        geo_ir_data.to_netcdf(
            output_folder / "geo_ir" / "geo_ir.nc",
            encoding=encoding
        )

    return 0


@click.command()
@click.argument("collocation_path")
@click.argument("output_folder")
@click.option("--include_geo", help="Include geo observations.", is_flag=True, default=False)
@click.option("--include_geo_ir", help="Include geo IR observations.", is_flag=True, default=False)
@click.option("--glob_pattern", help="Optional glob pattern to subsect input files.", default="*.nc")
def extract_evaluation_data(
        collocation_path: str,
        output_folder: str,
        include_geo: bool = False,
        include_geo_ir: bool = False,
        glob_pattern: str = "*.nc"
) -> int:
    """
    Extract evaluation data in COLLOCATION_PATH and write scenes to OUTPUT_FOLDER.
    """
    from speed.data.utils import extract_evaluation_data

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error(
            "'collocation_path' must point to an existing directory."
        )
        return 1

    files_native = sorted(list((collocation_path / "native").glob(glob_pattern)))
    files_gridded = sorted(list((collocation_path / "gridded").glob(glob_pattern)))

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

    for median_time in track(combined, description="Extracting evaluation data:"):
        extract_evaluation_data(
            times_gridded[median_time],
            times_native[median_time],
            output_folder,
            include_geo=include_geo,
            include_geo_ir=include_geo_ir
        )



cli.add_command(extract_data, name="extract_data")
cli.add_command(extract_training_data_spatial, name="extract_training_data_spatial")
cli.add_command(extract_training_data_tabular, name="extract_training_data_tabular")
cli.add_command(extract_evaluation_data, name="extract_evaluation_data")
cli.add_command(cpcir.cli, name="extract_cpcir_obs")
cli.add_command(goes.cli, name="extract_goes_obs")
