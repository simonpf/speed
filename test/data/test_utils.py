"""
Tests for the speed.data.utils module.
"""
import numpy as np
from speed.data.utils import get_smoothing_kernel


def test_get_smoothing_kernel():
    """
    Ensure that calulated smoothing kernel has:
        - Expected size of 3 time normalized FWHM
        - Maximum at center
        - Pixels within FWHM radius are larger than half maximum
        - Pixels outside FWHM radius are smaller than half maximum
    """
    k = get_smoothing_kernel(5, 1)

    assert k.shape == (15, 15)
    assert np.argmax(k[7]) == 7
    maximum = k[7, 7]
    assert k[7, 5] > 0.5 * maximum
    assert k[7, 4] < 0.5 * maximum
    assert k[5, 7] > 0.5 * maximum
    assert k[4, 7] < 0.5 * maximum
