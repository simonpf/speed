"""
spree.data.gpm_gv
===============

This module provides function to extract collocations with GPM ground-validation (GV) data.
"""
import logging
from typing import List, Optional, Union, Tuple
import warnings

import dask.array as da
import numpy as np
from pansat.products import Product
from pansat import FileRecord, TimeRange, Granule
from pansat.products.ground_based import gpm_gv
from pyresample.geometry import AreaDefinition, SwathDefinition
from pansat.catalog import Index
from scipy.signal import convolve
import xarray as xr

from speed.data.reference import ReferenceData
from speed.grids import GLOBAL
from speed.data.utils import (
    get_smoothing_kernel,
    extract_rect,
    calculate_footprint_averages,
    interp_along_swath,
    resample_data
)
from .mrms import (
    smooth_field,
    downsample_mrms_data,
    footprint_average_mrms_data
)


LOGGER = logging.getLogger(__name__)


def load_gv_data(
        granule: Granule,
        gv_products: Tuple[Product, Product, Product, Product]
) -> Union[xr.Dataset, None]:
    """
    Load all MRMS data corresponding to a single MRMS precip rate granule.

    Args:
        granule: A granule object defining a MRMS precip rate file.
        gv_products: A tuple containing the ground-validation data products.

    Return:
        An xarray.Dataset containing the surface precip rate, precipitation
        type, radar quality index, and gauge-correction factor.
    """
    input_files = []
    precip_rate_rec = granule.file_record
    precip_rate_rec = precip_rate_rec.get()
    time_range = precip_rate_rec.temporal_coverage

    precip_rate_prod, rqi_prod, precip_type_prod, gcf_prod = gv_products

    # Load and resample precip rate.
    precip_data = precip_rate_prod.open(precip_rate_rec)
    input_files.append(precip_rate_rec.filename)

    # Find and resample corresponding radar quality index data.
    rqi_recs = rqi_prod.get(precip_rate_rec.central_time)
    rqi_rec = precip_rate_rec.find_closest_in_time(rqi_recs)[0]
    if precip_rate_rec.time_difference(rqi_rec).total_seconds() > 0:
        return None
    rqi_data = rqi_prod.open(rqi_rec)
    input_files.append(rqi_rec.filename)

    # Find and resample precip type data.
    precip_type_recs = precip_type_prod.get(precip_rate_rec.central_time)
    precip_type_rec = precip_rate_rec.find_closest_in_time(precip_type_recs)[0]
    if precip_rate_rec.time_difference(precip_type_rec).total_seconds() > 0:
        return None
    precip_type_data = precip_type_prod.open(precip_type_rec)
    input_files.append(precip_type_rec.filename)

    # Find and resample gauge correction factors.
    precip_gcf_recs = gcf_prod.get(precip_rate_rec.central_time)
    precip_gcf_rec = precip_rate_rec.find_closest_in_time(precip_gcf_recs)[0]
    if precip_rate_rec.time_difference(precip_gcf_rec).total_seconds() > 0:
        LOGGER.warning(
            "Couldn't find matching gauge-corrected MRMS measurements for "
            "MRMS precip rate file '%s'.",
            precip_rate_rec.filename
        )
        return None
    gcf_data = gcf_prod.open(precip_gcf_rec).drop_vars(["time"])

    input_files.append(precip_gcf_rec.filename)

    data = xr.Dataset({
        "surface_precip": precip_data.precip_rate,
        "radar_quality_index": rqi_data.rqi,
        "precip_type": precip_type_data.mask,
        "gauge_correction_factor": gcf_data["1hcf"],
        "time": precip_data.time
    })

    data.attrs["input_files"] = input_files
    return data


class GPMGV(ReferenceData):
    """
    Reference data class for processing MRMS data.

    Combines MRMS precip rate, flag and radar quality index into
    a DataArray.
    """

    def __init__(self, name, gv_products):
        self.products = gv_products
        super().__init__(name, gpm_gv.MRMS_DOMAIN, self.products[0])

    def load_reference_data(
        self,
        input_granule: Granule,
        granules: List[Granule],
        beam_width: float
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of MRMS data.

        Args:
            input_granule: The granule of the input observations.
            granules: A list containing the matched reference data granules
            beam_width: The beam width to assume for calculating footprint
                averages.

        Return:
            An xarray.Dataset containing the GPM GV reference data.
        """
        coords = input_granule.geometry.bounding_box_corners
        lon_min, lat_min, lon_max, lat_max = coords

        ref_data = []

        col_start = None
        col_end = None
        row_start = None
        row_end = None

        input_files = []

        grid = None
        gv_data = []

        input_data = input_granule.open()
        if "latitude_s1" in input_data:
            input_data = input_data.rename(
                latitude_s1="latitude",
                longitude_s1="longitude"
            )
        input_data = input_data[[
            "scan_time",
            "latitude",
            "longitude",
            "spacecraft_latitude",
            "spacecraft_longitude",
            "spacecraft_altitude",
        ]]
        input_data = input_data.reset_coords(names=["scan_time"])
        lons_input = input_data.longitude.data
        lats_input = input_data.latitude.data

        for granule in list(granules):
            gv_data_t = load_gv_data(granule, self.products)
            input_files += gv_data_t.attrs["input_files"]
            if col_start is None:
                lons = gv_data_t.longitude.data
                lats = gv_data_t.latitude.data
                lons_input = lons_input[
                    (lons_input >= lons.min()) *
                    (lons_input <= lons.max())
                ]
                lats_input = lats_input[
                    (lats_input >= lats.min()) *
                    (lats_input <= lats.max())
                ]
                lon_min, lon_max = lons_input.min(), lons_input.max()
                lat_min, lat_max = lats_input.min(), lats_input.max()

                lon_indices = np.where(
                    (
                        (gv_data_t.longitude.data >= lon_min) *
                        (gv_data_t.longitude.data <= lon_max)
                    )
                )[0]
                lat_indices = np.where(
                    (
                        (gv_data_t.latitude.data >= lat_min) *
                        (gv_data_t.latitude.data <= lat_max)
                    )
                )[0]
                row_start = lat_indices.min()
                row_end = lat_indices.max()
                col_start = lon_indices.min()
                col_end = lon_indices.max()

            gv_data.append(
                gv_data_t[{
                    "latitude": slice(row_start, row_end),
                    "longitude": slice(col_start, col_end),
                }]
            )

        if len(gv_data) == 0:
            LOGGER.warning(
                "Unable to load complete GPM GV data for input graule %s.",
                input_granule
            )
            return None

        gv_data = xr.concat(gv_data, "time")

        lons = gv_data.longitude.data
        lats = gv_data.latitude.data
        lons, lats = np.meshgrid(lons, lats)

        area = SwathDefinition(lons=lons, lats=lats)

        scan_time, _ = xr.broadcast(input_data.scan_time, input_data.latitude)
        dtype = scan_time.dtype
        input_data["scan_time"] = scan_time.astype("int64")
        scan_time = resample_data(input_data, area)["scan_time"].astype(dtype)
        gv_data = interp_along_swath(gv_data.sortby("time"), scan_time, dimension="time")
        gv_data_d, _ = downsample_mrms_data(gv_data, grid=grid)

        latitudes = input_data.latitude if "latitude" in input_data else input_data.latitude_s1
        longitudes = input_data.longitude if "longitude" in input_data else input_data.longitude_s1
        sensor_latitude = input_data.spacecraft_latitude
        sensor_longitude = input_data.spacecraft_longitude
        sensor_altitude = input_data.spacecraft_altitude
        LOGGER.info(
            "Calculating footprint averages for input granule %s",
            input_granule
        )
        gv_data_fpavg = footprint_average_mrms_data(
            gv_data,
            longitudes,
            latitudes,
            scan_time,
            sensor_longitude,
            sensor_latitude,
            sensor_altitude,
            beam_width=beam_width,
            area_of_influence=1.0
        )
        gv_data_d.attrs["gpm_gv_input_files"] = input_files
        return gv_data_d, gv_data_fpavg


gv_data_gpm = GPMGV(
    "gpm_gv_gpm",
    (gpm_gv.precip_rate_gpm, gpm_gv.rqi_gpm, gpm_gv.mask_gpm, gpm_gv.gcf_gpm)
)
