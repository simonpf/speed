"""
speed.data.gpm
==============

This module contains the code to process GPM L1C data into SPEED collocations.
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
import pansat
from pansat import FileRecord, TimeRange, Granule
from pansat.environment import get_index
from pansat.products.satellite.gpm import (
    merged_ir,
    l1c_metopa_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_gcomw1_amsr2,
    l1c_tropics03_tms,
    l1c_tropics06_tms,
    l1c_r_gpm_gmi
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
    save_data_native,
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


def run_preprocessor(gpm_granule: Granule) -> xr.Dataset:
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
            l1c_file = extract_scans(gpm_granule, tmp, min_scans=128)
            os.chdir(tmp)
            sensor = L1CFile(l1c_file).sensor
            preprocessor_data = preprocessor.run_preprocessor(
                l1c_file, sensor, robust=False
            )
    finally:
        os.chdir(old_dir)

    preprocessor_data = preprocessor_data.rename({
        "scans": "scan",
        "pixels": "pixel",
        "channels": "channel_gprof",
        "brightness_temperatures": "observations_gprof",
        "earth_incidence_angle": "earth_incidence_angle_gprof"
    })
    invalid = preprocessor_data.observations_gprof.data < 0
    preprocessor_data.observations_gprof.data[invalid] = np.nan

    return preprocessor_data


def load_l1c_brightness_temperatures(
        gpm_granule: Granule,
        preprocessor_data: xr.Dataset,
        radius_of_influence: float
) -> np.ndarray:
    """
    Load and, if required, remap L1C brightness temperatures to preprocessor swath.

    Args:
        gpm_granule: The granule of GPM L1C data from which to extract the brightness temperatures.
        preprocessor_data: An xarray.Dataset containing the data from the preprocessor.
        radius_of_influence: The radius of influence to use for the nearest neighbor resampling
             of the brightness temperatures.

    Return:
        A numpy array containing all combined brightness temperatures from the L1C file.
    """
    l1c_data = gpm_granule.open()
    lats_p = preprocessor_data.latitude.data
    lons_p = preprocessor_data.longitude.data

    if "tbs" in l1c_data.variables:
        lats = l1c_data.latitude.data
        lons = l1c_data.longitude.data
        if lats.shape == lats_p.shape:
            return l1c_data.tbs

        source = SwathDefinition(lons=lons, lats=lats)
        target = SwathDefinition(lons=lons_p, lats=lats_p)
        tbs = kd_tree.resample_nearest(
            source,
            l1c_data.tbs.data,
            target,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        eia = kd_tree.resample_nearest(
            source,
            l1c_data.incidence_angle.data,
            target,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        if eia.ndim < tbs.ndim:
            eia = np.broadcast_to(eia[..., None], tbs.shape)
        return tbs, eia

    tbs = []
    eia = []
    swath = 1
    while f"latitude_s{swath}" in l1c_data.variables:
        lats = l1c_data[f"latitude_s{swath}"].data
        lons = l1c_data[f"longitude_s{swath}"].data

        source = SwathDefinition(lons=lons, lats=lats)
        target = SwathDefinition(lons=lons_p, lats=lats_p)
        tbs.append(
            kd_tree.resample_nearest(
                source,
                l1c_data[f"tbs_s{swath}"].data,
                target,
                radius_of_influence=radius_of_influence,
                fill_value=np.nan
            )
        )
        eia_s = kd_tree.resample_nearest(
            source,
            l1c_data[f"incidence_angle_s{swath}"].data,
            target,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        if eia_s.ndim < tbs[-1].ndim:
            eia_s = np.broadcast_to(eia_s[..., None], tbs[-1].shape)
        eia.append(eia_s)
        swath += 1

    tbs = np.concatenate(tbs, axis=-1)
    eia = np.concatenate(eia, axis=-1)
    invalid = tbs < 0
    tbs[invalid] = np.nan
    return tbs, eia


class GPMInput(InputData):
    """
    The GPMInput class provides functionality to load data from any sensor
    of the GPM constellation and combine this data with any of the reference
    data classes.
    """
    def __init__(
            self,
            name: str,
            products: List[pansat.Product],
            beam_width: float = 0.98,
            radius_of_influence: float = 60e3
    ):
        """
        Args:
            name: The name of the input data source, which is used to identify the data from
                the command line interface.
            products: The pansat products from which to obtain input data files.
            beam_width: The beam width to use for the calculation of footprint averages.
            radius_of_influence: The radius of influence to use for the resampling of the
                sensor data.
        """
        super().__init__(name)
        self.products = products
        self.beam_width = beam_width
        self.radius_of_influence = radius_of_influence

        characteristics_dir = Path(__file__).parent / "sensor_characteristics"
        with open(characteristics_dir / (name + ".toml")) as char_file:
            self.characteristics = toml.loads(char_file.read())

    def process_match(
        self,
        match: Tuple[Granule, List[Granule]],
        reference_data: ReferenceData,
        output_folder: Path,
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
        preprocessor_data.attrs.pop("frequencies")

        tbs_mw, eia_mw = load_l1c_brightness_temperatures(
            inpt_granule,
            preprocessor_data,
            self.radius_of_influence
        )
        preprocessor_data["observations"] = (("scan", "pixel", "channel"), tbs_mw.data)
        preprocessor_data["earth_incidence_angle"] = (("scan", "pixel", "channel"), eia_mw.data)
        preprocessor_data["observations"].attrs.update(self.characteristics["channels"])
        preprocessor_data["observations_gprof"].attrs.update(self.characteristics["channels_gprof"])

        # Load and combine reference data for all matche granules
        ref_data, ref_data_fpavg = reference_data.load_reference_data(
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
            latitude=preprocessor_data.latitude,
            longitude=preprocessor_data.longitude,
            method="nearest",
        )
        reference_data_r["time"] = ref_data.time.astype(np.int64).interp(
            latitude=preprocessor_data.latitude,
            longitude=preprocessor_data.longitude,
            method="nearest"
        ).astype("datetime64[ns]")

        for var in ref_data_fpavg:
            if var in reference_data_r:
                reference_data_r[var + "_fpavg"] = ref_data_fpavg[var]

        # Limit scans to scans with useful data.
        scan_start, scan_end = get_useful_scan_range(
            reference_data_r,
            "surface_precip",
            min_scans=256,
            margin=64
        )

        preprocessor_data = preprocessor_data[{"scan": slice(scan_start, scan_end)}]
        reference_data_r = reference_data_r[{"scan": slice(scan_start, scan_end)}]
        preprocessor_data.attrs["scan_start"] = inpt_granule.primary_index_range[0] + scan_start
        preprocessor_data.attrs["scan_end"] = inpt_granule.primary_index_range[0] + scan_end

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

        indices = calculate_grid_resample_indices(preprocessor_data, grid)
        preprocessor_data["row_index"] = indices.row_index
        preprocessor_data["col_index"] = indices.col_index

        gpm_input_file = inpt_granule.file_record.filename
        preprocessor_data.attrs["gpm_input_file"] = gpm_input_file

        # Save data in native format.
        LOGGER.info(
            "Saving file in native format to %s.",
            output_folder
        )
        time = save_data_native(
            self.name,
            reference_data.name,
            preprocessor_data,
            reference_data_r,
            output_folder,
            min_scans=128
        )

        # Save data in gridded format.
        preprocessor_data_r = resample_data(
            preprocessor_data, grid, self.radius_of_influence
        )
        for name, attr in preprocessor_data.attrs.items():
            preprocessor_data_r.attrs[name] = attr

        indices = calculate_swath_resample_indices(
            preprocessor_data, grid, self.radius_of_influence
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

            # Get all available files of GPM product for given day.
            gpm_recs = product.get(time_range)
            gpm_index = Index.index(product, gpm_recs)
            LOGGER.info(
                f"Found %s files for %s/%s/%s.",
                len(gpm_recs), year, month, day
            )

            # If reference data has a fixed domain, subset index to files
            # intersecting the domain.
            if reference_data.domain is not None:
                gpm_index = gpm_index.subset(roi=reference_data.domain)

            # Collect available reference data.
            reference_recs = []
            for granule in gpm_index.granules:
                    reference_recs += reference_data.pansat_product.get(
                        time_range=granule.time_range
                    )
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
                    self.process_match(match, reference_data, output_folder)
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

gmi = GPMInput("gmi", [l1c_r_gpm_gmi], beam_width=0.98, radius_of_influence=15e3)

AMSR2_PRODUCTS = [l1c_gcomw1_amsr2]
amsr2 = GPMInput("amsr2", AMSR2_PRODUCTS, radius_of_influence=6e3)
