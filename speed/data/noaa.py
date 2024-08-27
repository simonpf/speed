"""
speed.data.noaa
==============

This module provides functionality to extract collocations from NOAA GAASP
L1B files.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Tuple


from filelock import FileLock
import pansat
from pansat import FileRecord, TimeRange, Granule
from pansat.environment import get_index
from pansat.products.satellite.noaa.gaasp import (
    l1b_gcomw1_amsr2
)
from pansat.granule import merge_granules, Granule
from pansat.catalog import Index
from pansat.catalog.index import find_matches
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import toml

import numpy as np
import xarray as xr

from speed.data.utils import (
    extract_scans,
    save_data_on_swath,
    save_data_gridded,
    calculate_grid_resample_indices,
    calculate_swath_resample_indices,
    resample_data,
    extract_rect,
    get_useful_scan_range,
)
from speed.data.reference import ReferenceData
from speed.data.input import InputData
from speed import grids


LOGGER = logging.getLogger(__file__)




def load_brightness_temperatures(
        granule: Granule,
) -> np.ndarray:
    """
    Load AMSR2 brightness temperatures.

    Args:
        gpm_granule: The granule of GPM L1C data from which to extract the brightness temperatures.
        preprocessor_data: An xarray.Dataset containing the data from the preprocessor.
        radius_of_influence: The radius of influence to use for the nearest neighbor resampling
             of the brightness temperatures.

    Return:
        A numpy array containing all combined brightness temperatures from the L1C file.
    """
    l1b_data = granule.open()

    lats = l1b_data.latitude_s2.data
    lons = l1b_data.longitude_s2.data

    tbs_s1 = np.repeat(l1b_data.tbs_s1.data, 2, axis=1)
    tbs_s2 = l1b_data.tbs_s2.data
    tbs_s3 = l1b_data.tbs_s3.data
    sc_lon = l1b_data.spacecraft_longitude.data
    sc_lat = l1b_data.spacecraft_latitude.data
    sc_alt = l1b_data.spacecraft_altitude.data
    tbs = np.concatenate((tbs_s1, tbs_s2, tbs_s3), axis=-1)
    scan_time = l1b_data.scan_time.data

    # Upsample fields
    n_scans, n_pixels, n_channels = tbs.shape

    lons_hr = np.zeros_like(lons, shape=(2 * n_scans - 1, n_pixels))
    lons_hr[::2] = lons
    lons_hr[1::2] = 0.5 * (lons[:-1] + lons[1:])
    lats_hr = np.zeros_like(lats, shape=(2 * n_scans - 1, n_pixels))
    lats_hr[::2] = lats
    lats_hr[1::2] = 0.5 * (lats[:-1] + lats[1:])
    tbs_hr = np.zeros_like(tbs, shape=(2 * n_scans - 1, n_pixels, n_channels))
    tbs_hr[::2] = tbs
    tbs_hr[1::2] = 0.5 * (tbs[:-1] + tbs[1:])
    scan_time_hr = np.zeros_like(scan_time, shape=(2 * n_scans - 1))
    scan_time_hr[::2] = scan_time
    scan_time_hr[1::2] = scan_time[1:] + 0.5 * (scan_time[1:] - scan_time[:-1])

    sc_lon_hr = np.zeros_like(sc_lon, shape=(2 * n_scans - 1))
    sc_lon_hr[::2] = sc_lon
    sc_lon_hr[1::2] = 0.5 * (sc_lon[:-1] + sc_lon[1:])
    sc_lat_hr = np.zeros_like(sc_lat, shape=(2 * n_scans - 1))
    sc_lat_hr[::2] = sc_lat
    sc_lat_hr[1::2] = 0.5 * (sc_lat[:-1] + sc_lat[1:])
    sc_alt_hr = np.zeros_like(sc_alt, shape=(2 * n_scans - 1))
    sc_alt_hr[::2] = sc_alt
    sc_alt_hr[1::2] = 0.5 * (sc_alt[:-1] + sc_alt[1:])

    return xr.Dataset({
        "latitude": (("scan", "pixel"), lats_hr),
        "longitude": (("scan", "pixel"), lons_hr),
        "observations": (("scan", "pixel", "channels"), tbs_hr),
        "spacecraft_longitude": (("scan"), sc_lon_hr),
        "spacecraft_latitude": (("scan"), sc_lat_hr),
        "spacecraft_altitude": (("scan"), sc_alt_hr),
        "scan_time": (("scan"), scan_time_hr)
    })


class NOAAGAASPInput(InputData):
    """
    The NOAAGAASP class provides functionality to load data from NOAA AMSR2
    L1B files.
    """
    def __init__(
            self,
            name: str,
            products: List[pansat.Product],
            radius_of_influence: float = 5e3
    ):
        """
        Args:
            name: The name of the input data source, which is used to identify the data from
                the command line interface.
            products: The pansat products from which to obtain input data files.
        """
        super().__init__(name)
        self.products = products
        self.radius_of_influence = radius_of_influence
        self.beam_width = None


    def process_match(
        self,
        match: Tuple[Granule, List[Granule]],
        reference_data: ReferenceData,
        output_folder: Path
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
            if not scans.any():
                return None
            start_time = l1c_data.scan_time.data[scans].min()
            end_time = l1c_data.scan_time.data[scans].max()
            inpt_granule.time_range = TimeRange(start_time, end_time)

        ref_geom = ref_granules[0].geometry.to_shapely()
        inpt_geom = inpt_granule.geometry.to_shapely()
        area = ref_geom.intersection(inpt_geom).area
        if area < 15:
            LOGGER.info(
                "Discarding match between '%s' and '%s' because area "
                " is less than 25 square degree.",
                inpt_granule.file_record.product.name,
                reference_data.pansat_product.name,
            )
            return None

        inpt_data = load_brightness_temperatures(inpt_granule,)

        # Load and combine reference data for all matche granules
        ref_data, _ = reference_data.load_reference_data(
            inpt_granule,
            ref_granules,
            radius_of_influence=self.radius_of_influence,
            beam_width=self.beam_width
        )
        if ref_data is None:
            LOGGER.info(
                "No reference data for %s.",
                ref_granules
            )
            return None


        reference_data_r = ref_data.interp(
            latitude=inpt_data.latitude,
            longitude=inpt_data.longitude,
            method="nearest",
        )
        reference_data_r["time"] = ref_data.time.astype(np.int64).interp(
            latitude=inpt_data.latitude,
            longitude=inpt_data.longitude,
            method="nearest"
        ).astype("datetime64[ns]")

        # Limit scans to scans with useful data.
        scan_start, scan_end = get_useful_scan_range(
            reference_data_r,
            "surface_precip",
            min_scans=256,
            margin=64
        )

        for var in reference_data_r:
            reference_data_r[var].encoding = ref_data[var].encoding

        inpt_data = inpt_data[{"scan": slice(scan_start, scan_end)}]
        reference_data_r = reference_data_r[{"scan": slice(scan_start, scan_end)}]
        inpt_data.attrs["scan_start"] = inpt_granule.primary_index_range[0] + scan_start
        inpt_data.attrs["scan_end"] = inpt_granule.primary_index_range[0] + scan_end

        row_start = ref_data.attrs.get("lower_left_row", 0)
        n_rows = ref_data.latitude.size
        col_start = ref_data.attrs.get("lower_left_col", 0)
        n_cols = ref_data.longitude.size

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

        grid = grids.GLOBAL.grid[
            row_start:row_start + n_rows,
            col_start:col_start + n_cols
        ]

        indices = calculate_grid_resample_indices(inpt_data, grid)
        inpt_data["row_index"] = indices.row_index
        inpt_data["col_index"] = indices.col_index

        gpm_input_file = inpt_granule.file_record.filename
        inpt_data.attrs["gpm_input_file"] = gpm_input_file

        # Save data in on_swath format.
        LOGGER.info(
            "Saving file in on_swath format to %s.",
            output_folder
        )
        time = save_data_on_swath(
            self.name,
            reference_data.name,
            inpt_data,
            reference_data_r,
            output_folder,
            min_scans=128
        )

        # Save data in gridded format.
        inpt_data_r = resample_data(
            inpt_data, grid, self.radius_of_influence
        )
        for name, attr in inpt_data.attrs.items():
            inpt_data_r.attrs[name] = attr

        indices = calculate_swath_resample_indices(
            inpt_data, grid, self.radius_of_influence
        )
        ref_data["scan_index"] = indices.scan_index
        ref_data["pixel_index"] = indices.pixel_index

        LOGGER.info(
            "Saving file in gridded format to %s.",
            output_folder
        )
        save_data_gridded(
            self.name,
            reference_data.name,
            time,
            inpt_data_r,
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
        """
        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(days=1)
        time_range = TimeRange(start_time, end_time)

        for product in self.products:
            LOGGER.info(f"Starting processing of product '%s'.", product.name)

            # Get all available files for given day.
            inpt_recs = product.get(time_range)
            inpt_index = Index.index(product, inpt_recs)
            LOGGER.info(
                f"Found %s files for %s-%s-%s.",
                len(inpt_recs), year, f"{month:02}", f"{day:02}"
            )

            # If reference data has a fixed domain, subset index to files
            # intersecting the domain.
            if reference_data.domain is not None:
                inpt_index = inpt_index.subset(roi=reference_data.domain)

            # Collect available reference data.
            reference_recs = []
            lock = FileLock("noaa_ref.lock")
            with lock:
                for granule in inpt_index.granules:
                        reference_recs += reference_data.pansat_product.get(
                            time_range=granule.time_range
                        )
            reference_index = Index.index(reference_data.pansat_product, reference_recs)

            # Calculate matches between input and reference data.
            matches = find_matches(inpt_index, reference_index)

            LOGGER.info(
                f"Found %s matches for input data product '%s' and "
                f"reference data product '%s'.",
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


noaa_amsr2 = NOAAGAASPInput(
    "noaa_amsr2",
    [l1b_gcomw1_amsr2],
    radius_of_influence=5e3
)
