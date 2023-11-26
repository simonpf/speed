"""
spree.data.gpm
==============

This module contains the code to process GPM L1C data into SPREE collocations.
"""
from datetime import datetime, timedelta
import logging
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Optional

from gprof_nn.data.l1c import L1CFile
from gprof_nn.data import preprocessor

from pansat import FileRecord, TimeRange, Granule
from pansat.products.satellite.gpm import (
    merged_ir,
    l1c_metopa_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_gcomw1_amsr2,
)
from pansat.granule import merge_granules, Granule
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree

import numpy as np
import xarray as xr

from speed.data.utils import (
    extract_scans,
    save_data_native,
    save_data_gridded,
    calculate_grid_resample_indices,
    calculate_swath_resample_indices,
    resample_data,
    extract_rect,
)
from speed.data.reference import ReferenceData
from speed.data.input import InputData
from speed import grids


LOGGER = logging.getLogger(__file__)


def run_preprocessor(gpm_granule):
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
            preprocessor_data = preprocessor.run_preprocessor(
                l1c_file, sensor, robust=False
            )
    finally:
        os.chdir(old_dir)

    preprocessor_data = preprocessor_data.rename({"brightness_temperatures": "tbs_mw"})
    invalid = preprocessor_data.tbs_mw.data < 0
    preprocessor_data.tbs_mw.data[invalid] = np.nan

    return preprocessor_data


def interp_along_swath(ref_data, scan_time, dimension="time"):
    """
    Interpolate time-gridded data to swath.

    Helper function that interpolates time-gridded data to sensor
    scan times.

    Args:
        ref_data: An xarray.Dataset containing the data to interpolate.
        scan_time: An xarray.DataArray containing the scan times to which
            to interpolate 'ref_data'.
        dimension: The name of the time dimension along which to
            interpolate.

    Return:
        The interpolated dataset.
    """
    time_bins = ref_data.time.data
    time_bins = np.concatenate(
        [
            [time_bins[0] - 0.5 * (time_bins[1] - time_bins[0])],
            time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1]),
            [time_bins[-1] + 0.5 * (time_bins[-1] - time_bins[-2])],
        ],
        axis=0,
    )
    inds = np.digitize(scan_time.astype(np.int64), time_bins.astype(np.int64))
    inds = np.maximum(inds - 1, 0)
    inds = xr.DataArray(inds, dims=(("latitude", "longitude")))
    return ref_data[{"time": inds}]


class GPMInput(InputData):
    """
    Represents input data from a sensor type of the GPM constellation.
    """

    def __init__(self, name, products, radius_of_influence=60e3):
        super().__init__(name)
        self.products = products
        self.radius_of_influence = radius_of_influence

    def process_match(
        self,
        match: Tuple[Granule, List[Granule]],
        reference_data: ReferenceData,
        output_folder: Path,
        lock: None,
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
                collocations in native and gridded format.
            lock: multiprocessing.Lock to use to synchronize file downloads.
        """
        inpt_granule, ref_granules = match
        ref_granules = list(ref_granules)

        domain = None
        if hasattr(reference_data, "domain"):
            domain = reference_data.domain

        # Reduce time range of input granule to avoid loading
        # too much reference data
        if domain is not None:
            corners = domain.bounding_box_corners
            lon_min, lat_min, lon_max, lat_max = corners
            l1c_data = inpt_granule.file_record.product.open(inpt_granule.file_record)
            if "latitude_s1" in l1c_data:
                lons = l1c_data.longitude_s1.data
                lats = l1c_data.latitude_s1.data
            else:
                lons = l1c_data.longitude.data
                lats = l1c_data.latitude.data
            scans = (
                (lons > lon_min)
                * (lons < lon_max)
                * (lats > lat_min)
                * (lats < lat_max)
            ).any(-1)
            start_time = l1c_data.scan_time.data[scans].min()
            end_time = l1c_data.scan_time.data[scans].max()
            inpt_granule.time_range = TimeRange(start_time, end_time)

        ref_geom = ref_granules[0].geometry.to_shapely()
        gpm_geom = inpt_granule.geometry.to_shapely()
        area = ref_geom.intersection(gpm_geom).area
        if area < 25:
            LOGGER.info(
                "Discarding match between '%s' and '%s' because area "
                " is less than 25 square degree.",
                inpt_granule.file_record.product.name,
                reference_data.pansat_product.name,
            )
            return None

        preprocessor_data = run_preprocessor(inpt_granule)

        # Load and combine reference data for all matches granules
        ref_data = reference_data.load_reference_data(inpt_granule, ref_granules)
        if "time" in ref_data:
            reference_data_r = ref_data.interp(
                latitude=preprocessor_data.latitude,
                longitude=preprocessor_data.longitude,
                time=preprocessor_data.scan_time,
                method="nearest",
            )
        else:
            reference_data_r = ref_data.interp(
                latitude=preprocessor_data.latitude,
                longitude=preprocessor_data.longitude,
                method="nearest",
            )

        # Determine number of valid reference pixels in scene.
        if "surface_precip" in reference_data_r:
            surface_precip = reference_data_r.surface_precip.data
        else:
            surface_precip = reference_data_r.surface_precip_combined.data
        if np.isfinite(surface_precip).sum() < 50:
            LOGGER.info(
                "Skipping match because it contains only %s valid "
                " surface precip pixels.",
                np.isfinite(surface_precip).sum(),
            )

            return None

        indices = calculate_grid_resample_indices(preprocessor_data, grids.GLOBAL)
        preprocessor_data["row_index"] = indices.row_index
        preprocessor_data["col_index"] = indices.col_index

        # Add IR data.
        ir_time_range = inpt_granule.time_range.expand(timedelta(minutes=15))
        if lock is not None:
            lock.acquire()
        try:
            ir_files = merged_ir.get(time_range=ir_time_range)
        finally:
            if lock is not None:
                lock.release()
        ir_data = xr.concat(
            [xr.load_dataset(rec.local_path) for rec in ir_files], "time"
        )
        ir_data = ir_data.rename(
            {"lon": "longitude", "lat": "latitude", "Tb": "tbs_ir"}
        )
        tbs_ir = ir_data.tbs_ir.interpolate_na(dim="latitude", method="nearest")
        tbs_ir = tbs_ir.interp(
            latitude=preprocessor_data.latitude,
            longitude=preprocessor_data.longitude,
            time=preprocessor_data.scan_time,
        )
        preprocessor_data["tbs_ir"] = (("scans", "pixels"), tbs_ir.data)

        # Save data in native format.
        time = save_data_native(
            self.name,
            reference_data.name,
            preprocessor_data,
            reference_data_r,
            output_folder,
        )

        # Save data in gridded format.
        preprocessor_data_r = resample_data(
            preprocessor_data, grids.GLOBAL.grid, self.radius_of_influence
        )
        tbs_ir = ir_data.tbs_ir.interpolate_na(dim="latitude", method="nearest")
        preprocessor_data_r["tbs_ir"] = tbs_ir

        indices = calculate_swath_resample_indices(
            preprocessor_data, grids.GLOBAL, self.radius_of_influence
        )

        row_start = ref_data.attrs.get("lower_left_row", 0)
        n_rows = ref_data.latitude.size
        col_start = ref_data.attrs.get("lower_left_col", 0)
        n_cols = ref_data.longitude.size
        indices = indices[
            {
                "longitude": slice(col_start, col_start + n_cols),
                "latitude": slice(row_start, row_start + n_rows),
            }
        ]
        preprocessor_data_r = extract_rect(
            preprocessor_data_r,
            col_start,
            col_start + n_cols,
            row_start,
            row_start + n_rows,
        )
        ref_data["scan_index"] = indices.scan_index
        ref_data["pixel_index"] = indices.pixel_index

        if "time" in ref_data:
            scan_time = preprocessor_data_r.scan_time
            scan_time = scan_time.fillna(value=scan_time.min())
            ref_data = interp_along_swath(ref_data, scan_time, dimension="time")

        save_data_gridded(
            self.name,
            reference_data.name,
            time,
            preprocessor_data_r,
            ref_data,
            output_folder,
        )

    def process_day(
        self,
        year: int,
        month: int,
        day: int,
        reference_data: ReferenceData,
        output_folder: Path,
        lock: Optional[multiprocessing.Lock] = None,
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
            lock: Optional lock object to use to synchronize downloads.
        """
        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        for product in self.products:
            LOGGER.info(f"Starting processing of product '%s'.", product.name)

            # Get all available files of GPM product for given day.
            if lock is not None:
                lock.acquire()
            try:
                gpm_recs = product.get(time_range)
            finally:
                if lock is not None:
                    lock.release()
            gpm_index = Index.index(product, gpm_recs)
            # If reference data has a fixed domain, subset index to files
            # intersecting the domain.
            if reference_data.domain is not None:
                gpm_index = gpm_index.subset(roi=reference_data.domain)

            # Collect available reference data.
            reference_recs = []
            for granule in gpm_index.granules:
                if lock is not None:
                    lock.acquire()
                try:
                    reference_recs += reference_data.pansat_product.get(
                        time_range=granule.time_range
                    )
                finally:
                    if lock is not None:
                        lock.release()
            reference_index = Index.index(reference_data.pansat_product, reference_recs)

            # Calculate matches between input and reference data.
            matches = find_matches(gpm_index, reference_index)

            LOGGER.info(
                f"Found %s matches for input data product '%s' and "
                f"reference data product '%s'.",
                len(matches),
                product.name,
                reference_data.pansat_product.name,
            )

            for match in matches:
                try:
                    self.process_match(match, reference_data, output_folder, lock=lock)
                except Exception as exc:
                    LOGGER.warning(
                        "The following exception was encountered when processing "
                        " granule '%s'.",
                        match[0],
                    )
                    LOGGER.exception(exc)


MHS_PRODUCTS = [
    l1c_metopa_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
]

mhs = GPMInput("mhs", MHS_PRODUCTS, radius_of_influence=64e3)


AMSR2_PRODUCTS = [l1c_gcomw1_amsr2]

amsr2 = GPMInput("amsr2", AMSR2_PRODUCTS, radius_of_influence=64e3)
