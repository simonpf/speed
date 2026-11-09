"""
speed.data.earth_care
=====================

This module implements SPEED collocation extraction for EarthCARE overpasses.
"""
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

from filelock import FileLock
from gprof_nn.data.l1c import L1CFile
from gprof_nn.data import preprocessor
import pansat
from pansat import TimeRange, Granule
from pansat.environment import get_index
from pansat.products.satellite.earth_care import (
    l2b_ac_tc,
    l2b_acm_cap,
    l2b_acm_clp,
)
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import toml

import numpy as np
from scipy.stats import binned_statistic_2d
import xarray as xr

from pansat.utils import resample_data_binned, resample_data
from speed.data.utils import (
    extract_scans,
    save_data_on_swath,
    save_data_gridded,
    calculate_grid_resample_indices,
    calculate_swath_resample_indices,
    get_useful_scan_range,
    get_lon_lat_bins
)
from speed.data.reference import ReferenceData
from speed.data.input import InputData
from speed import grids


LOGGER = logging.getLogger(__file__)


def lowest_clutter_free_bin(data: xr.Dataset) -> np.ndarray:
    """
    Determine the index of the lowest clutter-free bin.

    Args:

    """
    mask = data.cpr_detection.data
    valid = np.flip(~np.isin(mask, [0, -1, 9, 10, 11, 12]), -1)
    inds = []
    for ind in range(valid.shape[0]):
        locs = np.where(valid[ind])[0]
        if 0 < len(locs):
            inds.append(np.min(locs))
        else:
            inds.append(-1)
    return np.array(inds)

class EarthCAREInput(InputData):
    """
    The EarthCAREInput class provides functionality to combine precipitation estimates from
    the EarthCARE mission with reference data.
    """
    def __init__(self):
        super().__init__("earth_care")

    def load_earth_care_data(self, ec_granule) -> xr.Dataset:
        """
        Load EarthCARE data from matched granule.
        """
        ec_data = ec_granule.open()

        time_range = ec_granule.time_range
        central_time = time_range.start + 0.5 * (time_range.end - time_range.start)
        cap_rec = l2b_acm_cap.get(central_time)
        if len(cap_rec) == 0:
            raise ValueError(
                "No ACM CAP record for granule %s.", ec_granule
            )

        lons_tc = ec_data.longitude.data
        lats_tc = ec_data.latitude.data
        swath = SwathDefinition(lons=lons_tc, lats=lats_tc)
        cap_data = resample_data(cap_rec[0].open(), swath, new_dims=("rays",))

        ec_data["rain_rate"] = cap_data["rain_rate"]
        ec_data["snowfall_rate"] = cap_data["snowfall_rate"]
        clutter_free_bin = lowest_clutter_free_bin(ec_data)

        lcb = lowest_clutter_free_bin(ec_data)
        print(ec_data.range_bins.size, np.unique(lcb))
        invalid = lcb < 0
        lcb = ec_data.range_bins.size - 1 - lcb
        lcb[invalid] = -1
        ec_data["lowest_clutter_free_bin"] = (("rays",), lcb)
        print(np.unique(lcb))

        lcb = ec_data["lowest_clutter_free_bin"]
        surface_precip = ec_data["rain_rate"][{"range_bins": lcb}]
        surface_precip.data[invalid] = np.nan
        ec_data["surface_precip"] = surface_precip
        surface_snow = ec_data["snowfall_rate"][{"range_bins": lcb}]
        surface_snow.data[invalid] = np.nan
        ec_data["surface_snow"] = surface_snow

        return ec_data

    def process_match(
        self,
        match: Tuple[Granule, List[Granule]],
        reference_data: ReferenceData,
        output_folder: Path,
        min_area: float = 0.0
    ) -> None:
        """
        Extract collocations from matched input and reference data.

        Args:
            match: A tuple ``(inpt_granule, output_granules)`` describing
                a match of a granule of input data and multiple granules of
                reference data.
            reference_data: The reference data object to use to load the
                reference data for the matched granules.
            output_folder: The base folder to which to write the extracted
                collocations in on_swath and gridded format.
            min_area: A minimum area in square degree below which collocations
                are discarded.
        """
        inpt_granule, ref_granules = match
        ref_granules = list(ref_granules)

        ec_data = self.load_earth_care_data(inpt_granule)

        ec_data.time.attrs = {}
        for var in ec_data.variables.values():
            var.attrs = {}

        ec_data.height.encoding = {
            "dtype": "float32",
            "zlib": True
        }
        ec_data.cpr_detection.encoding = {
            "dtype": "int16",
            "zlib": True
        }
        ec_data.synergetic_detection.encoding = {
            "dtype": "int16",
            "zlib": True
        }
        ec_data.lowest_clutter_free_bin.encoding = {
            "dtype": "int16",
            "zlib": True
        }
        ec_data.rain_rate.encoding = {
            "dtype": "float32",
            "zlib": True
        }
        ec_data.snowfall_rate.encoding = {
            "dtype": "float32",
            "zlib": True
        }


        # Load and combine reference data for all matche granules
        ref_data, ref_data_fpavg = reference_data.load_reference_data(
            inpt_granule,
            ref_granules,
            radius_of_influence=5e3,
            beam_width=None
        )
        if ref_data is None:
            LOGGER.info(
                "No reference data for %s.",
                ref_granules
            )
            return None

        reference_data_r = ref_data.interp(
            latitude=ec_data.latitude,
            longitude=ec_data.longitude,
            method="nearest",
        )
        reference_data_r["time"] = ref_data.time.astype(np.int64).interp(
            latitude=ec_data.latitude,
            longitude=ec_data.longitude,
            method="nearest"
        ).astype("datetime64[ns]")

        if ref_data_fpavg is not None:
            for var in ref_data_fpavg:
                if var in reference_data_r:
                    reference_data_r[var + "_fpavg"] = ref_data_fpavg[var]

        row_start = ref_data.attrs.get("lower_left_row", 0)
        n_rows = ref_data.latitude.size
        col_start = ref_data.attrs.get("lower_left_col", 0)
        n_cols = ref_data.longitude.size

        # Determine number of valid reference pixels in scene.
        if "surface_precip" in reference_data_r:
            surface_precip = reference_data_r.surface_precip.data
        else:
            surface_precip = reference_data_r.surface_precip_combined.data
        if np.isfinite(surface_precip).sum() < 1:
            LOGGER.info(
                "Skipping match because it contains only %s valid "
                " surface precip pixels.",
                np.isfinite(surface_precip).sum(),
            )
            return None

        grid = grids.GLOBAL.grid[
            row_start:row_start + n_rows,
            col_start:col_start + n_cols
        ]

        indices = calculate_grid_resample_indices(ec_data, grid)
        ec_data["row_index"] = indices.row_index
        ec_data["col_index"] = indices.col_index

        ec_input_file = inpt_granule.file_record.filename
        ec_data.attrs["ec_input_file"] = ec_input_file

        # Save data in on_swath format.
        LOGGER.info(
            "Saving file in on_swath format to %s.",
            output_folder
        )
        time = save_data_on_swath(
            self.name,
            reference_data.name,
            ec_data,
            reference_data_r,
            output_folder,
            min_scans=128
        )

        ec_data_r = resample_data_binned(ec_data, grid)
        LOGGER.info(
            "Saving file in gridded format to %s.",
            output_folder
        )
        save_data_gridded(
            self.name,
            reference_data.name,
            time,
            ec_data_r,
            ref_data,
            output_folder,
            min_size=256
        )

        del reference_data
        del reference_data_r
        del ec_data
        del ec_data_r

    def process_day(
        self,
        year: int,
        month: int,
        day: int,
        reference_data: ReferenceData,
        output_folder: Path,
        min_area: float = 5
    ):
        """
        Process all collocations available on a given day.

        Args:
            year: The year of the day's date.
            month: The month the day's date.
            day: The day of the month of the day's date.
            reference_data: A reference data object representing the
                reference data source.
            output_folder: The folder to which to write the extracted
                collocations.
            min_area: A minimum area in square degree below which collocations
                are discarded.
        """
        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        product = l2b_ac_tc
        LOGGER.info("Starting processing of product '%s'.", product.name)

        ref_recs = reference_data.pansat_product.find_files(time_range)
        if len(ref_recs) == 0:
            LOGGER.info(
                "No reference data found for %s.", time_range
            )

        if reference_data.domain is not None:
            inpt_recs = product.get(time_range, roi=reference_data.domain)
        else:
            inpt_recs = product.get(time_range)
        inpt_index = Index.index(product, inpt_recs)
        LOGGER.info(
            "Found %s files for %s/%s/%s.",
            len(inpt_recs), year, month, day
        )

        # Collect available reference data.
        reference_recs = []
        for granule in inpt_index.granules:
            reference_recs += reference_data.pansat_product.get(granule.time_range)
        LOGGER.info("Indexing reference data.")
        reference_index = Index.index(reference_data.pansat_product, reference_recs)

        # Calculate matches between input and reference data.
        LOGGER.info("Searching matches.")
        matches = find_matches(inpt_index, reference_index)

        LOGGER.info(
            "Found %s matches for input data product '%s' and "
            "reference data product '%s'.",
            len(matches),
            product.name,
            reference_data.pansat_product.name,
        )

        for match in matches:
            try:
                self.process_match(match, reference_data, output_folder)
            except Exception as exc:
                LOGGER.warning(
                    "The following exception was encountered when processing "
                    " granule '%s'.",
                    match[0],
                )
                LOGGER.exception(exc)


earth_care = EarthCAREInput()
