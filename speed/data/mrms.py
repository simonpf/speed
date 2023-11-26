"""
spree.data.mrms
===============

This module provides function to extract collocations with the NOAA
Multi-Radar Multi-Sensor (MRMS) ground-based radar estimates.
"""
from typing import List, Optional, Union, Tuple

import dask.array as da
import numpy as np
from pansat import FileRecord, TimeRange, Granule
from pansat.products.ground_based import mrms
from pansat.catalog import Index
from pyresample.bucket import BucketResampler
from scipy.signal import convolve
import xarray as xr

from speed.data.reference import ReferenceData
from speed.grids import GLOBAL
from speed.data.utils import get_smoothing_kernel, extract_rect


PRECIP_CLASSES = {
    0: "No precipitation",
    1: "Warm stratiform rain",
    3: "Snow",
    6: "Convection",
    7: "Hail",
    10: "Cool stratiform rain",
    91: "Tropical stratiform rain",
    96: "Tropical convective rain",
}


_RESAMPLER = None


def get_resampler(mrms_data: xr.Dataset) -> BucketResampler:
    """
    Get resampler for MRMS grids.

    Args:
        mrms_data: xarray.Dataset containing MRMS data.

    Return:
        A pyresample bucket resampler object that can be used to resample
        MRMS data to the global SPEED grid.
    """
    global _RESAMPLER
    if _RESAMPLER is not None:
        return _RESAMPLER
    lons = mrms_data.longitude.data
    lats = mrms_data.latitude.data
    lons, lats = np.meshgrid(lons, lats)

    _RESAMPLER = BucketResampler(GLOBAL.grid, da.from_array(lons), da.from_array(lats))
    return _RESAMPLER


def smooth_field(data: np.ndarray) -> np.ndarray:
    """
    Smooth field to 0.036 degree resolution.

    Args:
        data: The input MRMS array as 0.01 resolution.

    Return:
        A numpy.ndarray containing the smoothed arrary.
    """
    k = get_smoothing_kernel(0.036, 0.01)
    invalid = np.isnan(data)
    data_r = np.nan_to_num(data, nan=0.0, copy=True)

    # FFT-based convolution can produce negative values so remove
    # them here.
    data_s = np.maximum(convolve(data_r, k, mode="same"), 0.0)
    cts = np.maximum(convolve(~invalid, np.ones_like(k), mode="same"), 0.0)
    data_s = data_s / (cts / k.size)
    data_s[invalid] = np.nan
    return data_s


def resample_scalar(data: xr.DataArray) -> xr.DataArray:
    """
    Resamples scalar data (such as surface precip or RQI) using the
    bucket resampler.

    Args:
        data: A xarray.DataArray containing the data
            to resample.

    Return:
        An xr.DataArray containing the resampled data.
    """
    data_s = xr.DataArray(
        data=smooth_field(data.data),
        coords={"latitude": data.latitude, "longitude": data.longitude},
    )
    data_g = data_s.interp(latitude=GLOBAL.lats, longitude=GLOBAL.lons)
    return data_g


def resample_categorical(data: xr.DataArray, classes) -> xr.DataArray:
    """
    Resamples categorical data (such as the MRMS precip) using the
    bucket resampler.

    Args:
        data: A xarray.DataArray containing the data
            to resample.

    Return:
        A xr.DataArray containing the resampled data.
    """
    data_r = data.interp(
        latitude=GLOBAL.lats,
        longitude=GLOBAL.lons,
        method="nearest",
        kwargs={"fill_value": -3},
    )

    precip_class_repr = ""
    for value, name in PRECIP_CLASSES.items():
        precip_class_repr += f"{value} = {name}"
    data_r.attrs["classes"] = precip_class_repr

    return data_r


class MRMS(ReferenceData):
    """
    Reference data class for processing MRMS data.

    Combines MRMS precip rate, flag and radar quality index into
    a DataArray.
    """

    def __init__(self, name):
        super().__init__(name, mrms.MRMS_DOMAIN, mrms.precip_rate)

    def load_reference_data(
        self, input_granule: Granule, granules: List[Granule]
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of MRMS data.

        Args:
            granule: A granule object specifying the data to load.

        Return:
            An xarray.Dataset containing the MRMS reference data.
        """
        coords = input_granule.geometry.bounding_box_corners
        lon_min, lat_min, lon_max, lat_max = coords

        ref_data = []

        col_start = None
        col_end = None
        row_start = None
        row_end = None

        for granule in granules:
            precip_rate_rec = granule.file_record
            precip_rate_rec = precip_rate_rec.get()
            time_range = precip_rate_rec.temporal_coverage

            dims = ("latitude", "longitude")

            # Load and resample precip rate.
            precip_data = mrms.precip_rate.open(precip_rate_rec)
            lon_indices = (precip_data.longitude.data > lon_min) * (
                precip_data.longitude.data < lon_max
            )
            lat_indices = (precip_data.latitude.data > lat_min) * (
                precip_data.latitude.data < lat_max
            )
            precip_data = precip_data[
                {"longitude": lon_indices, "latitude": lat_indices}
            ]

            missing = precip_data.precip_rate.data < 0
            precip_data.precip_rate.data[missing] = np.nan
            precip_data_r = resample_scalar(precip_data.precip_rate)

            if col_start is None:
                valid = np.isfinite(precip_data_r.data)
                row_inds, col_inds = np.where(valid)
                col_start = col_inds.min()
                col_end = col_inds.max() + 1
                row_start = row_inds.min()
                row_end = row_inds.max()

            data_arrays = {
                "surface_precip": extract_rect(
                    precip_data_r, col_start, col_end, row_start, row_end
                )
            }

            # Find and resample corresponding radar quality index data.
            rqi_recs = mrms.radar_quality_index.get(precip_rate_rec.temporal_coverage)
            rqi_rec = precip_rate_rec.find_closest_in_time(rqi_recs)[0]
            if precip_rate_rec.time_difference(rqi_rec).total_seconds() > 0:
                return None
            else:
                rqi_data = mrms.radar_quality_index.open(rqi_rec)
                rqi_data = rqi_data[{"longitude": lon_indices, "latitude": lat_indices}]
                missing = rqi_data.radar_quality_index.data < 0
                rqi_data.radar_quality_index.data[missing] = np.nan
                rqi_data_r = resample_scalar(rqi_data.radar_quality_index)
                data_arrays["radar_quality_index"] = extract_rect(
                    rqi_data_r, col_start, col_end, row_start, row_end
                )

            # Find and resample precip flag data.
            precip_flag_recs = mrms.precip_flag.get(precip_rate_rec.temporal_coverage)
            precip_flag_rec = precip_rate_rec.find_closest_in_time(precip_flag_recs)[0]
            if precip_rate_rec.time_difference(precip_flag_rec).total_seconds() > 0:
                return None
            else:
                precip_flag_data = mrms.precip_flag.open(precip_flag_rec)
                precip_flag_data = precip_flag_data[
                    {"longitude": lon_indices, "latitude": lat_indices}
                ]
                precip_flag_data_r = resample_categorical(
                    precip_flag_data.precip_flag, PRECIP_CLASSES
                )
                data_arrays["precip_flag"] = extract_rect(
                    precip_flag_data_r, col_start, col_end, row_start, row_end
                )

            data = xr.Dataset(data_arrays)
            data["time"] = precip_data.time
            data.attrs = data_arrays["surface_precip"].attrs
            ref_data.append(data)

        ref_data = xr.concat(ref_data, "time").sortby("time")
        return ref_data


mrms_data = MRMS("mrms")
