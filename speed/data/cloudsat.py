"""
speed.data.cloudsat
===================

This module implements SPEED collocation extraction for CloudSat overpasses.
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
from pansat.products.satellite.cloudsat import (
    l2c_precip_column,
    l2c_rain_profile,
    l2c_snow_profile
)
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import toml

import numpy as np
from scipy.stats import binned_statistic_2d
import xarray as xr

from speed.data.utils import (
    extract_scans,
    save_data_on_swath,
    save_data_gridded,
    calculate_grid_resample_indices,
    calculate_swath_resample_indices,
    resample_data,
    get_useful_scan_range,
    get_lon_lat_bins
)
from speed.data.reference import ReferenceData
from speed.data.input import InputData
from speed import grids


LOGGER = logging.getLogger(__file__)




class CloudSatInput(InputData):
    """
    The CloudSatInput class provides functionality to combine precipitation estimates from
    the CloudSat mission with reference data.
    """
    def __init__(self):
        super().__init__("cloudsat")


    def load_cloudsat_data(self, cs_granule) -> xr.Dataset:
        """
        Load CloudSat data from matched granule.
        """
        cs_data = cs_granule.open().reset_coords("time")

        precip_column_rec = l2c_precip_column.get(cs_granule.time_range)
        if len(precip_column_rec) == 0:
            raise ValueError(
                "No 2C-PRECIP-COLUMN record for granule %s.", cs_granule
            )
        precip_column_data = l2c_precip_column.open(precip_column_rec[0])[{
            "rays": slice(*cs_granule.primary_index_range)
        }].reset_coords("time")
        cs_data["precip_flag"] = precip_column_data["precip_flag"]

        snow_profile_rec = l2c_snow_profile.get(cs_granule.time_range)
        if len(snow_profile_rec) == 0:
            raise ValueError(
                "No 2C-SNOW-PROFILE record for granule %s.", cs_granule
            )
        snow_profile_data = l2c_snow_profile.open(snow_profile_rec[0])[{
            "rays": slice(*cs_granule.primary_index_range)
        }].reset_coords("time")
        cs_data["surface_precip_snow"] = snow_profile_data["surface_precip"]

        pflag = cs_data["precip_flag"].data
        surface_precip = cs_data["surface_precip"].data
        surface_precip_snow = cs_data["surface_precip_snow"].data
        surface_precip_pc = precip_column_data["surface_precip"]
        total_precip = np.nan * np.zeros_like(surface_precip)
        total_precip[pflag == 0] = 0.0
        total_precip[surface_precip > 0] = surface_precip[surface_precip > 0]
        total_precip[surface_precip_snow > 0] = surface_precip_snow[surface_precip_snow > 0]
        cs_data["total_precip"] = (("rays",), total_precip)
        cs_data["surface_precip_pc"] = (("rays",), surface_precip_pc)
        cs_data["surface_snow_quality"] = (("rays",), surface_precip_snow["surface_snowfall_confidence"].data)
        cs_data = cs_data.assign_coords(scan_time=cs_data["time"])

        return cs_data

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

        cs_data = self.load_cloudsat_data(inpt_granule)

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
            latitude=cs_data.latitude,
            longitude=cs_data.longitude,
            method="nearest",
        )
        reference_data_r["time"] = ref_data.time.astype(np.int64).interp(
            latitude=cs_data.latitude,
            longitude=cs_data.longitude,
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

        #indices = calculate_grid_resample_indices(cs_data, grid)
        #cs_data["row_index"] = indices.row_index
        #cs_data["col_index"] = indices.col_index

        cs_input_file = inpt_granule.file_record.filename
        cs_data.attrs["cs_input_file"] = cs_input_file

        # Save data in on_swath format.
        LOGGER.info(
            "Saving file in on_swath format to %s.",
            output_folder
        )
        time = save_data_on_swath(
            self.name,
            reference_data.name,
            cs_data,
            reference_data_r,
            output_folder,
            min_scans=128
        )

        lons, lats = grid.get_lonlats()
        lon_bins, lat_bins = get_lon_lat_bins(lons, lats)
        lat_bins = np.flip(lat_bins)

        cs_data_r = {}
        for var in ["precip_flag", "surface_precip", "surface_precip_snow", "total_precip"]:
            vals = cs_data[var].data
            lats = cs_data.latitude.data
            lons = cs_data.longitude.data

            mask = np.isfinite(vals)
            if mask.sum() == 0:
                vals_r = np.nan * np.zeros((lon_bins.size - 1, lat_bins.size - 1))
            else:
                vals_r = binned_statistic_2d(
                    lons[mask], lats[mask], vals[mask], bins=(lon_bins, lat_bins)
                )[0][:, ::-1]
            cs_data_r[var] = (("latitude", "longitude"), vals_r.T)

        lat_bins = np.flip(lat_bins)
        lats = 0.5 * (lat_bins[1:] + lat_bins[:-1])
        lons = 0.5 * (lon_bins[1:] + lon_bins[:-1])
        cs_data_r['latitude'] = (('latitude',), lats)
        cs_data_r['longitude'] = (('longitude',), lons)
        cs_data_r = xr.Dataset(cs_data_r)

        LOGGER.info(
            "Saving file in gridded format to %s.",
            output_folder
        )
        save_data_gridded(
            self.name,
            reference_data.name,
            time,
            cs_data_r,
            ref_data,
            output_folder,
            min_size=256
        )

        del reference_data
        del reference_data_r
        del cs_data
        del cs_data_r


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

        product = l2c_rain_profile
        LOGGER.info("Starting processing of product '%s'.", product.name)

        ref_recs = reference_data.pansat_product.find_files(time_range)
        if len(ref_recs) == 0:
            LOGGER.info(
                "No reference data found for %s.", time_range
            )

        # Get all available files of CS product for given day.
        #lock = FileLock("cloudsat_inpt.lock")
        #with lock:
        #    cs_recs = product.get(time_range)
        cs_recs = product.get(time_range)
        cs_index = Index.index(product, cs_recs)
        LOGGER.info(
            "Found %s files for %s/%s/%s.",
            len(cs_recs), year, month, day
        )

        # If reference data has a fixed domain, subset index to files
        # intersecting the domain.
        if reference_data.domain is not None:
            cs_index = cs_index.subset(roi=reference_data.domain)
            LOGGER.info("Found %s granules over ROI.", len(cs_index.data))

        # Collect available reference data.
        reference_recs = []
        #with lock:
        for granule in cs_index.granules:
            reference_recs += reference_data.pansat_product.get(granule.time_range)
        LOGGER.info("Indexing reference data.")
        reference_index = Index.index(reference_data.pansat_product, reference_recs)

        # Calculate matches between input and reference data.
        LOGGER.info("Searching matches.")
        matches = find_matches(cs_index, reference_index)

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


cloudsat = CloudSatInput()
