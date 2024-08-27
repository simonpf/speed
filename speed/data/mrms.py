"""
speed.data.mrms
===============

This module provides functionality to extract collocations with the NOAA
Multi-Radar Multi-Sensor (MRMS) ground-based radar estimates.
"""
import logging
from typing import List, Optional, Union, Tuple
import warnings

import dask.array as da
import numpy as np
from pansat import FileRecord, TimeRange, Granule
from pansat.products.ground_based import mrms
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
    resample_data,
    interp_along_swath
)


LOGGER = logging.getLogger(__name__)


def smooth_field(data: np.ndarray) -> np.ndarray:
    """
    Smooth field to 0.036 degree resolution.

    Args:
        data: The input MRMS array as 0.01 resolution.

    Return:
        The input field smoothed using a Gaussien filter with a FWHM of 0.036 degree.
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


def resample_scalar(data: xr.DataArray) -> xr.DataArray:
    """
    Resamples scalar data (such as surface precip or RQI) using smoothing
    followed by linear interpolation.

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
    data_g = data_s.interp(latitude=GLOBAL.lats, longitude=GLOBAL.lons, method="nearest")
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


def load_mrms_data(granule: Granule) -> Union[xr.Dataset, None]:
    """
    Load all MRMS data corresponding to a single MRMS precip rate granule.

    Args:
        granule: A granule object defining a MRMS precip rate file.

    Return:
        An xarray.Dataset containing the surface precip rate, precipitation
        type, radar quality index, and gauge-correction factor.
    """
    input_files = []
    precip_rate_rec = granule.file_record
    precip_rate_rec = precip_rate_rec.get()
    time_range = precip_rate_rec.temporal_coverage

    # Load and resample precip rate.
    precip_data = mrms.precip_rate.open(precip_rate_rec)
    input_files.append(precip_rate_rec.filename)

    # Find and resample corresponding radar quality index data.
    rqi_recs = mrms.radar_quality_index.get(precip_rate_rec.central_time)
    rqi_rec = precip_rate_rec.find_closest_in_time(rqi_recs)[0]
    if precip_rate_rec.time_difference(rqi_rec).total_seconds() > 0:
        return None
    rqi_data = mrms.radar_quality_index.open(rqi_rec)
    input_files.append(rqi_rec.filename)

    # Find and resample precip flag data.
    precip_flag_recs = mrms.precip_flag.get(precip_rate_rec.central_time)
    precip_flag_rec = precip_rate_rec.find_closest_in_time(precip_flag_recs)[0]
    if precip_rate_rec.time_difference(precip_flag_rec).total_seconds() > 0:
        return None
    precip_flag_data = mrms.precip_flag.open(precip_flag_rec)
    input_files.append(precip_flag_rec.filename)

    # Find and resample gauge correction factors.
    precip_1h_gc_recs = mrms.precip_1h_gc.get(precip_rate_rec.central_time)
    if len(precip_1h_gc_recs) == 0:
        precip_1h_gc_recs = mrms.precip_1h_ms.get(precip_rate_rec.central_time)
    precip_1h_gc_rec = precip_rate_rec.find_closest_in_time(precip_1h_gc_recs)[0]
    if precip_rate_rec.time_difference(precip_1h_gc_rec).total_seconds() > 0:
        LOGGER.warning(
            "Couldn't find matching gauge-corrected MRMS measurements for "
            "MRMS precip rate file '%s'.",
            precip_rate_rec.filename
        )
        return None

    precip_1h_ro_recs = mrms.precip_1h.get(precip_1h_gc_rec.temporal_coverage.end)
    parts = precip_1h_gc_rec.filename.split("_")
    parts[0] = "RadarOnly"
    ro_filename = "_".join(parts)
    ro_filename = (
        precip_1h_gc_rec.filename
        .replace("MultiSensor", "RadarOnly")
        .replace("GaugeCorr", "RadarOnly")
        .replace("_Pass2", "")
    )
    precip_1h_ro_rec = [
        rec for rec in precip_1h_ro_recs if rec.filename == ro_filename
    ]
    if len(precip_1h_ro_rec) == 0:
        LOGGER.warning(
            "Couldn't find matching radar-only MRMS measurements for "
            "MRMS precip rate file '%s'.",
            precip_rate_rec.filename
        )
        return None
    precip_1h_ro_rec = precip_1h_ro_rec[0].get()
    precip_1h_ro_data = mrms.precip_1h.open(precip_1h_ro_rec).drop_vars(["time", "valid_time"], errors="ignore")
    precip_1h_gc_data = mrms.precip_1h_gc.open(precip_1h_gc_rec).drop_vars(["time", "valid_time"], errors="ignore")
    corr_factor_data = precip_1h_gc_data.precip_1h_gc / precip_1h_ro_data.precip_1h
    no_precip = np.isclose(precip_1h_ro_data.precip_1h.data, 0.0)
    corr_factor_data.data[no_precip] = 1.0
    invalid = precip_1h_gc_data.precip_1h_gc.data < 0.0
    corr_factor_data.data[invalid] = np.nan

    input_files.append(precip_1h_ro_rec.filename)
    input_files.append(precip_1h_gc_rec.filename)

    data = xr.Dataset({
        "surface_precip": precip_data.precip_rate * corr_factor_data,
        "radar_quality_index": rqi_data.radar_quality_index,
        "precip_type": precip_flag_data.precip_flag,
        "gauge_correction_factor": corr_factor_data
    })

    data.attrs["input_files"] = input_files
    return data


def downsample_mrms_data(
        mrms_data: xr.Dataset,
        grid: Optional[AreaDefinition] = None
) -> xr.Dataset:
    """
    Downsample MRMS dat to 0.036 resolution.

    The downsampling also converts the precip flag types to occurrences fractions for rain,
    snow, hail, convective and stratiform precipitation.

    Args:
        mrms_data: A xarray.Dataset containing the combining MRMS data for a given time step.
        grid: The grid to which to resample the data.


    Return:
        A new xarray.Dataset containing the downsampled MRMS data.

    """

    valid_mask = (
        (mrms_data["surface_precip"].data >= 0.0) *
        (mrms_data["precip_type"].data >= 0) *
        (mrms_data["radar_quality_index"].data >= 0.0) *
        np.isfinite(mrms_data["gauge_correction_factor"].data)
    )

    lower_left_row = None
    lower_left_col = None
    if grid is None:
        lons, lats = GLOBAL.grid.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]
        lons_ref = mrms_data.longitude.data
        lats_ref = mrms_data.latitude.data
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

    valid_fraction = smooth_field(valid_mask.astype(np.float32))

    surface_precip = mrms_data["surface_precip"].data
    surface_precip[~valid_mask] = np.nan
    surface_precip = smooth_field(surface_precip)

    gauge_correction_factor = mrms_data["gauge_correction_factor"].data
    gauge_correction_factor[~valid_mask] = np.nan
    gauge_correction_factor = smooth_field(gauge_correction_factor)

    radar_quality_index = mrms_data["radar_quality_index"].data
    radar_quality_index[~valid_mask] = np.nan
    radar_quality_index = smooth_field(radar_quality_index)

    precip_fraction = (mrms_data["precip_type"].data > 0).astype(np.float32)
    precip_fraction[~valid_mask] = np.nan
    precip_fraction = smooth_field(precip_fraction)

    snow_fraction = (
        (mrms_data["precip_type"].data == 3.0) |
        (mrms_data["precip_type"].data == 4.0)
    ).astype(np.float32)
    snow_fraction[~valid_mask] = np.nan
    snow_fraction = smooth_field(snow_fraction).astype(np.float32)

    hail_fraction = (mrms_data["precip_type"].data == 7.0).astype(np.float32)
    hail_fraction[~valid_mask] = np.nan
    hail_fraction = smooth_field(hail_fraction).astype(np.float32)

    conv_fraction = (
        (mrms_data["precip_type"].data == 6.0) |
        (mrms_data["precip_type"].data == 96.0)
    ).astype(np.float32)
    conv_fraction[~valid_mask] = np.nan
    conv_fraction = smooth_field(conv_fraction).astype(np.float32)

    strat_fraction = (
        (mrms_data["precip_type"].data == 1.0) |
        (mrms_data["precip_type"].data == 2.0) |
        (mrms_data["precip_type"].data == 10.0) |
        (mrms_data["precip_type"].data == 91.0)
    ).astype(np.float32)
    strat_fraction[~valid_mask] = np.nan
    strat_fraction = smooth_field(strat_fraction).astype(np.float32)

    smoothed = xr.Dataset({
        "latitude": (("latitude",), mrms_data.latitude.data.astype(np.float32)),
        "longitude": (("longitude",), mrms_data.longitude.data.astype(np.float32)),
        "surface_precip": (("latitude", "longitude"), surface_precip.astype(np.float32)),
        "surface_precip_nn": (("latitude", "longitude"), mrms_data.surface_precip.data.astype(np.float32)),
        "gauge_correction_factor": (("latitude", "longitude"), gauge_correction_factor.astype(np.float32)),
        "gauge_correction_factor_nn": (("latitude", "longitude"), mrms_data.gauge_correction_factor.data.astype(np.float32)),
        "radar_quality_index": (("latitude", "longitude"), radar_quality_index.astype(np.float32)),
        "radar_quality_index_nn": (("latitude", "longitude"), mrms_data.radar_quality_index.data.astype(np.float32)),
        "valid_fraction": (("latitude", "longitude"), valid_fraction.astype(np.float32)),
        "precip_fraction": (("latitude", "longitude"), precip_fraction.astype(np.float32)),
        "snow_fraction": (("latitude", "longitude"), snow_fraction.astype(np.float32)),
        "convective_fraction": (("latitude", "longitude"), conv_fraction.astype(np.float32)),
        "stratiform_fraction": (("latitude", "longitude"), strat_fraction.astype(np.float32)),
        "hail_fraction": (("latitude", "longitude"), hail_fraction.astype(np.float32)),
        "time": (("latitude", "longitude"), mrms_data.time.data.astype("datetime64[ns]"))
    })

    lons, lats = grid.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]
    smoothed["time"] = smoothed.time.astype(np.int64)
    downsampled = smoothed.interp(latitude=lats, longitude=lons, method="nearest")
    if lower_left_col is not None:
        downsampled.attrs["lower_left_col"] = lower_left_col
        downsampled.attrs["lower_left_row"] = lower_left_row
    downsampled["time"] = downsampled["time"].astype("datetime64[ns]")
    return downsampled, grid


def footprint_average_mrms_data(
        mrms_data: xr.Dataset,
        longitudes: xr.DataArray,
        latitudes: xr.DataArray,
        scan_time: xr.DataArray,
        sensor_longitudes: xr.DataArray,
        sensor_latitudes: xr.DataArray,
        sensor_altitudes: xr.DataArray,
        beam_width: float,
        area_of_influence: float
) -> xr.Dataset:
    """
    Downsample MRMS data to sensor resolution using downsampling.

    Args:
        mrms_data: A xarray.Dataset containing the combining MRMS data for a given time step.
        longitudes: A xarray.DataAttary containing the longitude coordinates to which to resample
            'data'.
        latitudes: A xarray.DataArray containing the latitude coordinates to which to resample
            'data'.
        sensor_longitudes: An xarray.DataArray containing the longitude coordinates of the sensor position
            corresponding to all observed pixels.
        sensor_latitudes: An xarray.DataArray containing the latitude coordinates of the sensor position
            corresponding to all observed pixels.
        beam_width: The beam width of the sensor.
        area_of_influence: The extent of the are in degree to consider for the footprint averaging.


    Return:
        A new xarray.Dataset containing the downsampled MRMS data downsampled to the snsor footprints.
    """
    scan_time, _ = xr.broadcast(scan_time, latitudes)

    valid_mask = (
        (mrms_data["surface_precip"].data >= 0.0) *
        (mrms_data["precip_type"].data >= 0) *
        (mrms_data["radar_quality_index"].data >= 0.0) *
        np.isfinite(mrms_data["gauge_correction_factor"].data)
    )
    surface_precip = mrms_data["surface_precip"].data
    surface_precip[~valid_mask] = np.nan
    gauge_correction_factor = mrms_data["gauge_correction_factor"].data
    gauge_correction_factor[~valid_mask] = np.nan
    radar_quality_index = mrms_data["radar_quality_index"].data
    radar_quality_index[~valid_mask] = np.nan

    precip_fraction = (mrms_data["precip_type"].data > 0).astype(np.float32)
    precip_fraction[~valid_mask] = np.nan

    snow_fraction = (
        (mrms_data["precip_type"].data == 3.0) +
        (mrms_data["precip_type"].data == 4.0)
    ).astype(np.float32)
    snow_fraction[~valid_mask] = np.nan

    hail_fraction = (mrms_data["precip_type"].data == 7.0).astype(np.float32)
    hail_fraction[~valid_mask] = np.nan

    conv_fraction = (
        (mrms_data["precip_type"].data == 6.0) +
        (mrms_data["precip_type"].data == 96.0)
    ).astype(np.float32)
    conv_fraction[~valid_mask] = np.nan

    strat_fraction = (
        (mrms_data["precip_type"].data == 1.0) +
        (mrms_data["precip_type"].data == 2.0) +
        (mrms_data["precip_type"].data == 10.0) +
        (mrms_data["precip_type"].data == 91.0)
    ).astype(np.float32)
    strat_fraction[~valid_mask] = np.nan

    data = xr.Dataset({
        "latitude": (("latitude",), mrms_data.latitude.data),
        "longitude": (("longitude",), mrms_data.longitude.data),
        "surface_precip": (("latitude", "longitude"), surface_precip),
        "gauge_correction_factor": (("latitude", "longitude"), gauge_correction_factor),
        "radar_quality_index": (("latitude", "longitude"), radar_quality_index),
        "valid_fraction": (("latitude", "longitude"), valid_mask.astype(np.float32)),
        "precip_fraction": (("latitude", "longitude"), precip_fraction),
        "snow_fraction": (("latitude", "longitude"), snow_fraction),
        "convective_fraction": (("latitude", "longitude"), conv_fraction),
        "stratiform_fraction": (("latitude", "longitude"), strat_fraction),
        "hail_fraction": (("latitude", "longitude"), hail_fraction),
        "time": mrms_data.time
    })

    data_fpavg = calculate_footprint_averages(
            data,
            longitudes,
            latitudes,
            sensor_longitudes,
            sensor_latitudes,
            sensor_altitudes,
            beam_width,
            area_of_influence=area_of_influence,
    )
    return data_fpavg



class MRMS(ReferenceData):
    """
    Reference data class for processing MRMS data.

    Combines MRMS precip rate, flag and radar quality index into
    a DataArray.
    """

    def __init__(self, name):
        super().__init__(name, mrms.MRMS_DOMAIN, mrms.precip_rate)

    def load_reference_data(
        self,
        input_granule: Granule,
        granules: List[Granule],
        radius_of_influence: float,
        beam_width: float
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of MRMS data.

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
        coords = input_granule.geometry.bounding_box_corners
        lon_min, lat_min, lon_max, lat_max = coords

        ref_data = []

        col_start = None
        col_end = None
        row_start = None
        row_end = None

        input_files = []

        grid = None
        mrms_data = []

        input_data = input_granule.open()
        if "latitude_s1" in input_data:
            input_data = input_data.rename(
                scans="scan",
                pixels_s2="pixel",
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

        for granule in list(granules):
            mrms_data_t = load_mrms_data(granule)
            if mrms_data_t is None:
                continue
            input_files += mrms_data_t.attrs["input_files"]
            if col_start is None:
                lons = mrms_data_t.longitude.data
                lats = mrms_data_t.latitude.data
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
                        (mrms_data_t.longitude.data >= lon_min) *
                        (mrms_data_t.longitude.data <= lon_max)
                    )
                )[0]
                lat_indices = np.where(
                    (
                        (mrms_data_t.latitude.data >= lat_min) *
                        (mrms_data_t.latitude.data <= lat_max)
                    )
                )[0]
                row_start = lat_indices.min()
                row_end = lat_indices.max()
                col_start = lon_indices.min()
                col_end = lon_indices.max()

            mrms_data.append(
                mrms_data_t[{
                    "latitude": slice(row_start, row_end),
                    "longitude": slice(col_start, col_end),
                }]
            )

        if len(mrms_data) == 0:
            LOGGER.warning(
                "Unable to load complete MRMS data for input graule %s.",
                input_granule
            )
            return None

        mrms_data = xr.concat(mrms_data, "time")

        lons = mrms_data.longitude.data
        lats = mrms_data.latitude.data
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
        mrms_data = interp_along_swath(
            mrms_data.sortby('time'),
            scan_time,
            dimension="time"
        )
        LOGGER.info(
            "Downsampling MRMS data for input granule %s",
            input_granule
        )
        mrms_data_d, _ = downsample_mrms_data(mrms_data, grid=grid)

        # Don't calculate footprint averages if beam width is None
        if beam_width is None:
            return mrms_data_d, None

        latitudes = input_data.latitude if "latitude" in input_data else input_data.latitude_s1
        longitudes = input_data.longitude if "longitude" in input_data else input_data.longitude_s1
        sensor_latitude = input_data.spacecraft_latitude
        sensor_longitude = input_data.spacecraft_longitude
        sensor_altitude = input_data.spacecraft_altitude
        LOGGER.info(
            "Calculating footprint averages for input granule %s",
            input_granule
        )
        mrms_data_fpavg = footprint_average_mrms_data(
            mrms_data,
            longitudes,
            latitudes,
            scan_time,
            sensor_longitude,
            sensor_latitude,
            sensor_altitude,
            beam_width=beam_width,
            area_of_influence=1.0
        )
        mrms_data_d.attrs["mrms_input_files"] = input_files
        return mrms_data_d, mrms_data_fpavg




mrms_data = MRMS("mrms")
