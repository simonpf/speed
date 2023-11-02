"""
spree.data.gpm
==============

This module contains the code to process GPM L1C data into SPREE collocations.
"""
from datetime import datetime, timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from gprof_nn.data.l1c import L1CFile
from gprof_nn.data import preprocessor, mirs

from pansat import FileRecord, TimeRange, Granule
from pansat.products.satellite.gpm import (
    l1c_metopb_mhs,
    l1c_metopc_mhs,
)
from pansat.catalog import Index
from pansat.environment import get_index

def extract_scans(
        granule: Granule,
        dest: Path
) -> Path:
    """
    Extract and write scans from L1C file into a separate file.

    Args:
        granule: A pansat granule specifying a subset of an orbit.
        dest: A directory to which the extracted scans will be written.

    Return:
        The path of the file containing the extracted scans.
    """
    scan_start, scan_end = granule.primary_index_range
    l1c_path = granule.file_record.local_path
    l1c_file = L1CFile(granule.file_record.local_path)
    output_filename = dest / l1c_path.name
    l1c_file.extract_scan_range(scan_start, scan_end, output_filename)
    return output_filename


def run_preprocessor(
        gpm_granule
):
    """
    Run preprocessor on a GPM granule.

    Args:
        gpm_granule: A pansat granule identifying a subset of an orbit
            of GPM L1C files.

    Return:
        An xarray.Dataset containing the results from the preprocessor.
    """
    old_dir = os.getcwd()

    try:
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            l1c_file = extract_scans(gpm_granule, tmp)
            os.chdir(tmp)
            sensor = L1CFile(l1c_file).sensor
            preprocessor_data = preprocessor.run_preprocessor(l1c_file, sensor)
    finally:
        os.chdir(old_dir)

    return preprocessor_data


def run_mirs(
        gpm_granule
):
    """
    Run MIRS retrieval on a GPM granule.

    Args:
        gpm_granule: A pansat granule identifying a subset of an orbit
            of GPM L1C files.

    Return:
        An xarray.Dataset containing the results from the mirs retrieval.
    """
    old_dir = os.getcwd()
    try:
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            l1c_file = extract_scans(gpm_granule, tmp)
            os.chdir(tmp)
            sensor = L1CFile(l1c_file).sensor
            mirs_results = mirs.run_mirs_retrieval(l1c_file, sensor)
    finally:
        os.chdir(old_dir)

    return preprocessor_data


class GPMInput:
    """
    Represents input data from a sensor type of the GPM constellation.
    """
    def __init__(
            self,
            name,
            products
    ):
        self.name = name
        self.products = products


    def process_day(self, year, month, day, reference_data):

        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        for prod in self.products:


            if hasattr(reference_data, "domain"):
                gpm_recs = prod.get(time_range, roi=reference_data.domain)
                ref_product = reference_data.pansat_product

                gpm_index = Index.index(prod, gpm_recs)
                granules = gpm_index.find(roi=reference_data.domain)

                ref_recs = []
                for granule in granules:
                    ref_recs += ref_product.get(time_range=granule.time_range)

                preprocessor_data = run_preprocessor(granule)
                return preprocessor_data


mhs = GPMInput("mhs", [l1c_metopb_mhs])
