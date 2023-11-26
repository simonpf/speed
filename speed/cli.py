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
def extract_data(input_data, reference_data, output_folder, year, month, days):
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

    pool = ProcessPoolExecutor(max_workers=8)
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    tasks = {}
    for day in days:
        task = pool.submit(
            input_dataset.process_day,
            year,
            month,
            day,
            reference_dataset,
            output_folder,
            lock=lock,
        )
        tasks[task] = (year, month, day)

    for task in as_completed(tasks):
        try:
            task.result()
        except Exception as exc:
            LOGGER.exception(exc)


cli.add_command(extract_data)
