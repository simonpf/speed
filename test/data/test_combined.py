"""
Tests for the speed.data.combined module.
=========================================

"""
from speed.data.combined import run_mirs, gpm_cmb


def test_run_mirs(gmi_granule):
    """
    Run MIRS retrieval and ensure that:
       - Expected variables are in returned dataset.
    """

    results_mirs = run_mirs(gmi_granule)
    assert "surface_precip_mirs" in results_mirs
    assert "chi_squared_mirs" in results_mirs
    assert "quality_flag_mirs" in results_mirs


def test_load_reference_data(cmb_match):
    """
    Test extraction of reference data from GPM CMB and ensure that
    - the returned dataset contains 'surface_precip' and 'surface_precip_mirs'
      variables.
    """
    input_granule, cmb_granules = cmb_match
    ref_data = gpm_cmb.load_reference_data(input_granule, cmb_granules)

    assert "surface_precip" in ref_data
    assert "surface_precip_mirs" in ref_data
