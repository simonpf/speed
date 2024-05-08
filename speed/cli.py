"""
speed.cli
=========

The command line interface of SPEED.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from pathlib import Path
from typing import List
from tqdm import tqdm
import xarray as xr

import click

import speed.logging

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


cli.add_command(extract_data)


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
@click.option(
    "--filename_pattern",
    type=str,
    default="collocation_{time}",
)
def extract_training_data_2d(
        collocation_path: str,
        output_folder: str,
        overlap: float = 0.0,
        size: int = 256,
        filename_pattern: str = "collocation_{time}",
        min_input_frac: float = None
) -> int:
    """
    Extract collocations for a given date.
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
    for collocation_file in tqdm(collocation_files):
        input_data = xr.load_dataset(collocation_file, group="input_data")
        reference_data = xr.load_dataset(collocation_file, group="reference_data")
        extract_scenes(
            input_data,
            reference_data,
            output_folder,
            overlap=overlap,
            size=size,
            filename_pattern=filename_pattern
        )

    return 1

cli.add_command(extract_training_data_2d)
