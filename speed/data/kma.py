"""
speed.data.kma
==============

This module provides functionality to extract collocations with the KMA
ground-based radar data.
"""
import logging
from typing import List, Optional
import warnings

import numpy as np
from pansat.products.ground_based import kma
from pansat.granule import Granule
from scipy.signal import convolve
from pyresample.geometry import SwathDefinition

from speed.data.reference import ReferenceData
from speed.grids import GLOBAL
from speed.data.utils import (
    get_smoothing_kernel,
    resample_data,
    interp_along_swath
)

import xarray as xr


LOGGER = logging.getLogger(__name__)


def smooth_field(data: np.ndarray) -> np.ndarray:
    """
    Smooth field to approximate 0.036 degree resolution.

    Args:
        data: The input KMA data array at 1 km.

    Return:
        The input field smoothed using a Gaussian filter with a FWHM of 0.036 degree.
    """
    k = get_smoothing_kernel(0.036, 0.01)
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


def extract_collocation_data(kma_data: xr.Dataset) -> xr.Dataset:
    """
    Loads KMA data ensuring that all fields are set that are present
    in MRMS data.

    Args:
        granule: A granule object defining a KMA precip rate file.

    Return:
        An xarray.Dataset containing the surface precip rate, precipitation
        type, radar quality index, and gauge-correction factor.
    """
    surface_precip = kma_data.surface_precip.data
    surface_precip[surface_precip < 0.0] = np.nan

    surface_precip = smooth_field(kma_data.surface_precip)
    radar_quality_index = (0.0 <= surface_precip).astype(np.float32)
    precip_type = (surface_precip > 0).astype(np.float32)
    gauge_correction_factor = np.ones_like(surface_precip)
    valid_fraction = smooth_field((kma_data.surface_precip >= 0).astype(np.float32))
    precip_fraction = smooth_field((kma_data.surface_precip > 0).astype(np.float32))
    snow_fraction = np.zeros_like(precip_fraction)
    convection_fraction = np.zeros_like(precip_fraction)
    stratiform_fraction = np.zeros_like(precip_fraction)
    hail_fraction = np.zeros_like(precip_fraction)


    dims = (("y", "x"))
    data = xr.Dataset({
        "time": kma_data.time.data,
        "latitude": (dims, kma_data.latitude.data),
        "longitude": (dims, kma_data.longitude.data),
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


class KMAData(ReferenceData):
    """
    Reference data class for processing KMA ground-radar data.
    """
    def __init__(self):
        super().__init__("kma", kma.get_domain(), kma.precip_rate)

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
            )
        else:
            input_data = input_data.rename(
                scans="scan",
                pixels="pixel",
            )
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
            "spacecraft_altitude"
        ]]
        input_data = input_data.reset_coords(names=["scan_time"])
        lons_input = input_data.longitude.data
        lats_input = input_data.latitude.data

        col_start = None
        col_end = None
        row_start = None
        row_end = None

        kma_data = []

        for granule in list(granules):
            kma_data_t = granule.open()
            input_files.append(granule.file_record.filename)

            if col_start is None:
                lons = kma_data_t.longitude.data
                lats = kma_data_t.latitude.data
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

                row_inds, col_inds = np.where(
                    (
                        (kma_data_t.longitude.data >= lon_min) *
                        (kma_data_t.longitude.data <= lon_max) *
                        (kma_data_t.latitude.data >= lat_min) *
                        (kma_data_t.latitude.data <= lat_max)
                    )
                )
                row_start = row_inds.min()
                row_end = row_inds.max()
                col_start = col_inds.min()
                col_end = col_inds.max()


            kma_data.append(
                extract_collocation_data(
                    kma_data_t[{
                        "y": slice(row_start, row_end),
                        "x": slice(col_start, col_end),
                    }]
                )
            )

        if len(kma_data) == 0:
            LOGGER.warning(
                "Unable to load complete KMA data for input graule %s.",
                input_granule
            )
            return None

        kma_data = xr.concat(kma_data, "time")

        lons = kma_data.longitude.data[0]
        lats = kma_data.latitude.data[0]
        area = SwathDefinition(lons=lons, lats=lats)

        scan_time, _ = xr.broadcast(input_data.scan_time, input_data.latitude)
        input_data["scan_time"] = scan_time.astype("int64")
        scan_time = resample_data(
            input_data,
            area,
            radius_of_influence=radius_of_influence
        )["scan_time"]
        # Interpolation sets out of bounds values to -9999.
        invalid = scan_time < 0
        scan_time = scan_time.astype("datetime64[ns]")
        scan_time.data[invalid] = np.datetime64("NaT")

        kma_data = interp_along_swath(
            kma_data.sortby('time'),
            scan_time,
            dimension="time",
            ref_dims=(("y", "x"))
        )

        lons, lats = GLOBAL.grid.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]
        lons_ref = kma_data.longitude.data
        lats_ref = kma_data.latitude.data
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

        kma_data = kma_data.reset_coords(["time"])
        kma_data["time"] = kma_data.time.astype(np.int64)
        kma_data = resample_data(kma_data, grid)
        time = kma_data["time"]
        invalid = time < 0
        kma_data["time"] = kma_data.time.astype("datetime64[ns]")
        kma_data["time"].data[invalid] = np.datetime64("NaT")
        kma_data.attrs["lower_left_col"] = lon_start
        kma_data.attrs["lower_left_row"] = lat_start
        kma_data.attrs["input_files"] = ",".join(input_files)

        LOGGER.info(
            "Downsampling KMA data for input granule %s",
            input_granule
        )

        # Don't calculate footprint averages if beam width is None
        if beam_width is None:
            return kma_data, None

        raise ValueError(
            "Calculation of footprint averages for KMA data "
            "currently isn't supported."
        )



kma_data = KMAData()
