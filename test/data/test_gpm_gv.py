"""
Tests for the speed.data.gpm_gv module.
=====================================
"""
import pytest

import numpy as np
from pansat import TimeRange
from pansat.products.ground_based import mrms
import pansat.environment as penv

from speed.grids import GLOBAL
from speed.data.gpm_gv import (
    load_gv_data,
    gv_data_gpm
)


def test_load_gpm_gv_data(gpm_gv_match):
    """
    Ensure that loading of GPM GV data for a given precip rate granules fetches
    all of the expected additional data.
    """
    _, gv_granules = gpm_gv_match
    data = load_gv_data(next(iter(gv_granules)), gv_data_gpm.products)

    assert "surface_precip" in data
    assert "radar_quality_index" in data
    assert "precip_type" in data
    assert "gauge_correction_factor" in data

    assert np.any(data["surface_precip"].data >= 0.0)
    assert np.any(data["radar_quality_index"].data >= 0.0)
    assert np.any(data["precip_type"].data >= 0.0)
    assert np.any(data["gauge_correction_factor"].data >= 0.0)


def test_load_reference_data(gpm_gv_match):
    """
    Ensure that loading of reference data from multiple granules works.
    """
    input_granule, gpm_gv_granules = gpm_gv_match
    reference_data, reference_data_fpavg = gv_data_gpm.load_reference_data(
        input_granule,
        gpm_gv_granules,
        beam_width=0.98
    )
    assert "surface_precip" in reference_data
    assert np.any(reference_data["surface_precip"] >= 0.0)
    assert "radar_quality_index" in reference_data
    assert np.any(reference_data["radar_quality_index"] >= 0.0)
    assert "gauge_correction_factor" in reference_data
    assert np.any(reference_data["gauge_correction_factor"] >= 0.0)
    assert "surface_precip" in reference_data_fpavg
    assert "radar_quality_index" in reference_data_fpavg
    assert "gauge_correction_factor" in reference_data_fpavg
