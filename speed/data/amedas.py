"""
speed.data.amedas
=================

This module provides functionality to extract collocations with the AMeDAS
ground-based radar data.
"""
import logging
from typing import List, Optional, Union, Tuple
import warnings

import numpy as np
from pansat.products.ground_based import amedas
from pansat.geometry import LonLatRect
from pansat.granule import Granule
from scipy.signal import convolve
from pyresample.geometry import SwathDefinition

from speed.data.reference import ReferenceData
from speed.grids import GLOBAL
from speed.data.utils import (
    get_smoothing_kernel,
    extract_rect,
    calculate_footprint_averages,
    resample_data,
    interp_along_swath
)

import xarray as xr


LOGGER = logging.getLogger(__name__)


def smooth_field(data: np.ndarray) -> np.ndarray:
    """
    Smooth field to 0.036 degree resolution.

    Args:
        data: The input AMeDAS array at 0.0125 resolution.

    Return:
        The input field smoothed using a Gaussien filter with a FWHM of 0.036 degree.
    """
    k = get_smoothing_kernel(0.036, 0.0125)
    invalid = np.isnan(data)
    data_r = np.nan_to_num(data, nan=0.0, copy=True)

    # FFT-based convolution can produce negative values so remove
    # them here.
    data_s = np.maximum(convolve(data_r, k, mode="same"), 0.0)
    cts = np.maximum(convolve(~invalid, np.ones_like(k), mode="same"), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_s = data_s * (np.ones_like(k).sum() / cts)
    data_s[invalid] = np.nan
    return data_s


def extract_collocation_data(amedas_data: xr.Dataset) -> xr.Dataset:
    """
    Loads AMeDAS data ensuring that all fields are set that are present
    in MRMS data.

    Args:
        granule: A granule object defining a MRMS precip rate file.

    Return:
        An xarray.Dataset containing the surface precip rate, precipitation
        type, radar quality index, and gauge-correction factor.
    """
    surface_precip = smooth_field(amedas_data.pev)
    radar_quality_index = np.isfinite(surface_precip).astype(np.float32)
    precip_type = (surface_precip > 0).astype(np.float32)
    gauge_correction_factor = np.ones_like(surface_precip)
    valid_fraction = smooth_field((amedas_data.pev >= 0).astype(np.float32))
    precip_fraction = smooth_field((amedas_data.pev > 0).astype(np.float32))
    snow_fraction = np.zeros_like(precip_fraction)
    convection_fraction = np.zeros_like(precip_fraction)
    stratiform_fraction = np.zeros_like(precip_fraction)
    hail_fraction = np.zeros_like(precip_fraction)

    lower_left_row = None
    lower_left_col = None

    dims = (("latitude", "longitude"))
    data = xr.Dataset({
        "time": (("time",), amedas_data.time.data),
        "latitude": (("latitude",), amedas_data.latitude.data[:, 0]),
        "longitude": (("longitude",), amedas_data.longitude.data[0]),
        "surface_precip": (dims, surface_precip),
        "gauge_correction_factor": (dims, gauge_correction_factor),
        "radar_quality_index": (dims, radar_quality_index),
        "precip_type": (dims, precip_type),
        "valid_fraction": (dims, valid_fraction),
        "precip_fraction": (dims, precip_fraction),
        "snow_fraction": (dims, snow_fraction),
        "convection_fraction": (dims, convection_fraction),
        "stratiform_fraction": (dims, stratiform_fraction),
        "hail_fraction": (dims, hail_fraction)
    })
    return data


class AMeDASData(ReferenceData):
    """
    Reference data class for processing AMeDAS data.

    Combines AMeDAS precip rate, flag and radar quality index into
    a DataArray.
    """

    def __init__(self):
        super().__init__("amedas", amedas.AMEDAS_DOMAIN, amedas.precip_rate)

    def load_reference_data(
        self,
        input_granule: Granule,
        granules: List[Granule],
        radius_of_influence: float,
        beam_width: float
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of input data.

        Args:
            input_granule: The matched input data granules.
            granules: A list containing the matched reference data granules.
            radius_of_influence: The radius of influence to use for the resampling
                of the scan times.
            beam_width: The beam width to use for the calculation of footprint-averaged
                precipitation.

        Return:
            An xarray.Dataset containing the MRMS reference data.
        """
        input_files = []

        input_data = input_granule.open()
        if "pixels_s1" in input_data:
            input_data = input_data.rename(
                scans="scan",
                pixels_s1="pixel",
                latitude_s1="latitude",
                longitude_s1="longitude"
            )
        else:
            input_data = input_data.rename(
                scans="scan",
                pixels="pixel",
            )

        input_data = input_data[[
            "scan_time",
            "latitude",
            "longitude",
            "spacecraft_latitude",
            "spacecraft_longitude",
            "spacecraft_altitude"
        ]]
        input_data = input_data.reset_coords(names=["scan_time"])
        lons_input = input_data.longitude.data
        lats_input = input_data.latitude.data

        col_start = None
        col_end = None
        row_start = None
        row_end = None

        amedas_data = []

        for granule in list(granules):
            amedas_data_t = granule.open()
            input_files += granule.file_record.filename

            if col_start is None:
                lons = amedas_data_t.longitude.data
                lats = amedas_data_t.latitude.data
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
                        (amedas_data_t.longitude.data >= lon_min) *
                        (amedas_data_t.longitude.data <= lon_max)
                    )
                )[0]
                lat_indices = np.where(
                    (
                        (amedas_data_t.latitude.data >= lat_min) *
                        (amedas_data_t.latitude.data <= lat_max)
                    )
                )[0]
                row_start = lat_indices.min()
                row_end = lat_indices.max()
                col_start = lon_indices.min()
                col_end = lon_indices.max()


            amedas_data.append(
                extract_collocation_data(
                    amedas_data_t[{
                        "latitude": slice(row_start, row_end),
                        "longitude": slice(col_start, col_end),
                    }]
                )
            )

        if len(amedas_data) == 0:
            LOGGER.warning(
                "Unable to load complete AMeDAS data for input graule %s.",
                input_granule
            )
            return None

        amedas_data = xr.concat(amedas_data, "time")

        lons = amedas_data.longitude.data
        lats = amedas_data.latitude.data
        lons, lats = np.meshgrid(lons, lats)

        area = SwathDefinition(lons=lons, lats=lats)

        scan_time, _ = xr.broadcast(input_data.scan_time, input_data.latitude)
        dtype = scan_time.dtype
        input_data["scan_time"] = scan_time.astype("int64")
        scan_time = resample_data(
            input_data,
            area,
            radius_of_influence=radius_of_influence
        )["scan_time"].astype(dtype)
        amedas_data = interp_along_swath(
            amedas_data.sortby('time'),
            scan_time,
            dimension="time"
        )

        lons, lats = GLOBAL.grid.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]
        lons_ref = amedas_data.longitude.data
        lats_ref = amedas_data.latitude.data
        lon_min = lons_ref.min()
        lon_max = lons_ref.max()
        lat_min = lats_ref.min()
        lat_max = lats_ref.max()
        valid_lons = np.where((lons >= lon_min) * (lons <= lon_max))[0]
        valid_lats = np.where((lats >= lat_min) * (lats <= lat_max))[0]
        lon_start = max(valid_lons.min() - 64, 0)
        lon_end = min(valid_lons.max() + 64, lons.size - 1)
        lat_start = max(valid_lats.min() - 64, 0)
        lat_end = min(valid_lats.max() + 64, lats.size - 1)
        grid = GLOBAL.grid[lat_start:lat_end, lon_start:lon_end]
        lower_left_row = lat_start
        lower_left_col = lon_start
        amedas_data["time"] = amedas_data.time.astype(np.int64)
        amedas_data = amedas_data.interp(
            latitude=lats[lat_start:lat_end],
            longitude=lons[lon_start:lon_end],
            method="nearest"
        )
        amedas_data["time"] = amedas_data.time.astype("datetime64[ns]")
        amedas_data.attrs["lower_left_col"] = lon_start
        amedas_data.attrs["lower_left_row"] = lat_start

        LOGGER.info(
            "Downsampling AMeDAS data for input granule %s",
            input_granule
        )

        # Don't calculate footprint averages if beam width is None
        if beam_width is None:
            return amedas_data, None

        raise ValueError(
            "Calculation of footprint averages for AMeDAS data "
            "currently isn't supported."
        )



amedas = AMeDASData()
