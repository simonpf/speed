"""
spree.data.mrms
===============

This module provides function to extract collocations with the NOAA
Multi-Radar Multi-Sensor (MRMS) ground-based radar estimates.
"""
from typing import Union

import dask.array as da
import numpy as np
from pansat import FileRecord, TimeRange, Granule
from pansat.products.ground_based import mrms
from pyresample.bucket import BucketResampler
from scipy.signal import convolve
import xarray as xr

from speed.data.reference import ReferenceData
from speed.grids import GLOBAL
from speed.data.utils import get_smoothing_kernel


PRECIP_CLASSES = {
    0: "No precipitation",
    1: "Warm stratiform rain",
    3: "Snow",
    6: "Convection",
    7: "Hail",
    10: "Cool stratiform rain",
    91: "Tropical stratiform rain",
    96: "Tropical convective rain"
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

    _RESAMPLER = BucketResampler(
        GLOBAL.grid,
        da.from_array(lons),
        da.from_array(lats)
    )
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
    data_s = data_s / cts
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
        coords={
            "latitude": data.latitude,
            "longitude": data.longitude
        }
    )
    data_g = data_s.interp(
        latitude=GLOBAL.lats,
        longitude=GLOBAL.lons
    )
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
        kwargs={"fill_value": -3}
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
    def __init__(self):
        super().__init__(mrms.MRMS_DOMAIN, mrms.precip_rate)


    def load_reference_data(self, granule: Granule):
        """
        Load reference data for a given granule of MRMS data.

        Args:
            granule: A granule object specifying the data to load.

        Return:
            An xarray.Dataset containing the MRMS reference data.
        """
        precip_rate_rec = granule.file_record
        time_range = precip_rate_rec.temporal_coverage
        dims = ("latitude", "longitude")


        # Load and resample precip rate.
        precip_data = mrms.precip_rate.open(precip_rate_rec)
        missing = precip_data.precip_rate.data < 0
        precip_data.precip_rate.data[missing] = np.nan
        precip_data_r = resample_scalar(precip_data.precip_rate)
        datasets = {
            "surface_precip": (dims, precip_data_r.data)
        }

        # Find and resample corresponding radar quality index data.
        rqi_recs = mrms.radar_quality_index.get(precip_rate_rec.temporal_coverage)
        rqi_rec = precip_rate_rec.find_closest_in_time(rqi_recs)[0]
        if precip_rate_rec.time_difference(rqi_rec).total_seconds() > 0:
            return None
        else:
            rqi_data = mrms.radar_quality_index.open(rqi_rec)
            missing = rqi_data.radar_quality_index.data < 0
            rqi_data.radar_quality_index.data[missing] = np.nan
            rqi_data_r  = resample_scalar(rqi_data.radar_quality_index)
            datasets["radar_quality_index"] = (dims, rqi_data_r.data)


        # Find and resample precip flag data.
        precip_flag_recs = mrms.precip_flag.get(precip_rate_rec.temporal_coverage)
        precip_flag_rec = precip_rate_rec.find_closest_in_time(precip_flag_recs)[0]
        if precip_rate_rec.time_difference(precip_flag_rec).total_seconds() > 0:
            return None
        else:
            precip_flag_data = mrms.precip_flag.open(precip_flag_rec)
            precip_flag_data_r = resample_categorical(
                precip_flag_data.precip_flag,
                PRECIP_CLASSES
            )
            datasets["precip_flag"] = (dims, precip_flag_data_r.data)

        data = xr.Dataset(datasets)
        return data


mrms_data = MRMS()
