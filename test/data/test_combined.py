"""
Tests for the speed.data.combined module.
=========================================

"""
import os

import pytest

from speed.data.combined import run_mirs, gpm_cmb

try:
    from gprof_nn.data import mirs
    HAS_MIRS = True
except ImportError:
    HAS_MIRS = False


if "PANSAT_PASSWORD" in os.environ:
    HAS_PANSAT_PASSWORD = True
else:
    HAS_PANSAT_PASSWORD = False



@pytest.mark.skipif(not HAS_MIRS, reason="MIRS not available.")
@pytest.mark.skipif(not HAS_PANSAT_PASSWORD, reason="pansat password not set.")
def test_run_mirs(gmi_granule):
    """
    Run MIRS retrieval and ensure that:
       - Expected variables are in returned dataset.
    """

    results_mirs = run_mirs(gmi_granule)
    assert "surface_precip_mirs" in results_mirs
    assert "chi_squared_mirs" in results_mirs
    assert "quality_flag_mirs" in results_mirs


@pytest.mark.skipif(not HAS_PANSAT_PASSWORD, reason="pansat password not set.")
def test_load_reference_data(cmb_match):
    """
    Test extraction of reference data from GPM CMB and ensure that
    - the returned dataset contains 'surface_precip' and 'surface_precip_mirs'
      variables.
    """
    input_granule, cmb_granules = cmb_match
    ref_data = gpm_cmb.load_reference_data(input_granule, cmb_granules)

    assert "surface_precip" in ref_data
    assert "rain_water_content" in ref_data
