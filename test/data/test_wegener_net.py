"""
Tests for the speed.data.combined module.
=========================================
"""
import os

import pytest

import numpy as np

from speed.data.wegener_net import wegener_net


if "PANSAT_PASSWORD" in os.environ:
    HAS_PANSAT_PASSWORD = True
else:
    HAS_PANSAT_PASSWORD = False


@pytest.mark.skipif(not HAS_PANSAT_PASSWORD, reason="pansat password not set.")
def test_load_reference_data(wegener_net_match):
    """
    Test extraction of reference data from WegenerNet data and ensure that
    the precipitation field contains valid pixels.
    """
    input_granule, wn_granules = wegener_net_match
    ref_data, _ = wegener_net.load_reference_data(
        input_granule,
        wn_granules,
        5e3,
        0.98
    )

    assert "surface_precip" in ref_data
    assert np.any(np.isfinite(ref_data.surface_precip))
