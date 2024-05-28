"""
Tests for the speed.data.mrms module.
=====================================
"""
import pytest

import numpy as np
from pansat import TimeRange
from pansat.products.ground_based import mrms
import pansat.environment as penv

from speed.grids import GLOBAL
from speed.data.mrms import (
    load_mrms_data,
    downsample_mrms_data,
    footprint_average_mrms_data,
    mrms_data
)


def test_load_mrms_data(mrms_match):
    """
    Ensure that loading of MRMS data for a given precip rate granules fetches
    all of the expected additional data.
    """
    _, mrms_granules = mrms_match
    data = load_mrms_data(next(iter(mrms_granules)))

    assert "surface_precip" in data
    assert "radar_quality_index" in data
    assert "precip_type" in data
    assert "gauge_correction_factor" in data


def test_downsample_mrms_data(mrms_match):
    """
    Ensure that downsampling of MRMS data produces the expected variables and that the results
    contain some valid data.
    """
    _, mrms_granules = mrms_match
    data = load_mrms_data(next(iter(mrms_granules)))
    data_d, grid = downsample_mrms_data(data)

    assert "surface_precip" in data_d
    assert "surface_precip_nn" in data_d

    assert "valid_fraction" in data_d
    assert "rain_fraction" in data_d
    assert "snow_fraction" in data_d
    assert "convective_fraction" in data_d
    assert "stratiform_fraction" in data_d
    assert "convective_fraction" in data_d
    assert "stratiform_fraction" in data_d
    assert "hail_fraction" in data_d

    assert (data_d["surface_precip"] >= 0.0).any()
    assert (data_d["surface_precip_nn"] >= 0.0).any()
    assert (data_d["valid_fraction"] > 0.0).any()
    assert (data_d["stratiform_fraction"] > 0.0).any()


def test_footprint_average_mrms_data(mrms_match):
    """
    Ensure that footprint-averaging of MRMS data produces the expected variables and that the results
    contain some valid data.
    """
    input_granule, mrms_granules = mrms_match
    data = load_mrms_data(next(iter(mrms_granules)))

    input_data = input_granule.open()
    latitudes = input_data.latitude_s1
    longitudes = input_data.longitude_s1
    sensor_latitudes = input_data.spacecraft_latitude
    sensor_longitudes = input_data.spacecraft_longitude
    sensor_altitudes = input_data.spacecraft_altitude
    scan_time = input_data.scan_time

    data_fpavg = footprint_average_mrms_data(
        data,
        longitudes,
        latitudes,
        scan_time,
        sensor_longitudes,
        sensor_latitudes,
        sensor_altitudes,
        0.98,
        1.0
    )

    assert "surface_precip" in data_fpavg

    assert "valid_fraction" in data_fpavg
    assert "rain_fraction" in data_fpavg
    assert "snow_fraction" in data_fpavg
    assert "convective_fraction" in data_fpavg
    assert "stratiform_fraction" in data_fpavg
    assert "convective_fraction" in data_fpavg
    assert "stratiform_fraction" in data_fpavg
    assert "hail_fraction" in data_fpavg

    assert (data_fpavg["surface_precip"] >= 0.0).any()
    assert (data_fpavg["valid_fraction"] > 0.0).any()
    assert (data_fpavg["stratiform_fraction"] > 0.0).any()


def test_load_reference_data(mrms_match):
    """
    Ensure that loading of reference data from multiple granules works.
    """
    input_granule, mrms_granules = mrms_match
    reference_data, reference_data_fpavg = mrms_data.load_reference_data(
        input_granule,
        mrms_granules
    )
    assert "surface_precip" in reference_data
    assert "radar_quality_index" in reference_data
    assert "gauge_correction_factor" in reference_data

    assert "surface_precip" in reference_data_fpavg
    assert "radar_quality_index" in reference_data_fpavg
    assert "gauge_correction_factor" in reference_data_fpavg
