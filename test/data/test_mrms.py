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
    PRECIP_CLASSES,
    resample_scalar,
    resample_categorical,
    mrms_data
)


@pytest.fixture
def mrms_precip_rate():
    time = TimeRange("2019-01-01T00:00:00", "2019-01-01T00:00:00")
    recs = mrms.precip_rate.get(time)
    return recs


@pytest.fixture
def mrms_precip_flag():
    time = TimeRange("2019-01-01T00:00:00", "2019-01-01T00:00:00")
    recs = mrms.precip_flag.get(time)
    return recs

@pytest.fixture
def mrms_granule(mrms_precip_rate):
    index = penv.get_index(mrms.precip_rate)
    time = TimeRange("2019-01-01T00:00:00", "2019-01-01T00:00:00")
    return index.find(time)[0]


def test_resample_scalar(mrms_precip_rate):
    """
    Test resampling of MRMS scalar data and ensure that:
         - Resampled data has same shape as global grid.
    """
    rec = mrms_precip_rate[0]
    mrms_data = mrms.precip_rate.open(rec)
    resampled = resample_scalar(mrms_data.precip_rate)

    assert resampled.latitude.shape == GLOBAL.lats.shape
    assert resampled.longitude.shape == GLOBAL.lons.shape


def test_resample_categorical(mrms_precip_flag):
    """
    Test resampling of MRMS precip type data and ensure that:
         - Resampled data has same shape as global grid.
         - Resampled
    """
    rec = mrms_precip_flag[0]
    mrms_data = mrms.precip_flag.open(rec)
    resampled = resample_categorical(
        mrms_data.precip_flag,
        PRECIP_CLASSES
    )

    assert resampled.latitude.shape == GLOBAL.lats.shape
    assert resampled.longitude.shape == GLOBAL.lons.shape
    labels = np.unique(resampled.data)
    expected_labels = set([-3]).union(set(PRECIP_CLASSES.keys()))
    for label in labels:
        assert label in expected_labels


def test_load_mrms_data(mrms_match):
    """
    Test loading of MRMS data and ensure that the results dataset
        - Contains a 'surface_precip' variable
        - Contains a 'precip_flag' variable
        - Contains a 'radar_quality_index' variable
    """
    input_granule, mrms_granules = mrms_match
    reference_data = mrms_data.load_reference_data(
        input_granule,
        mrms_granules
    )

    assert "surface_precip" in reference_data
    assert "precip_flag" in reference_data
    assert "radar_quality_index" in reference_data
    assert "gauge_correction" in reference_data
