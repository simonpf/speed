"""
speed.data.utils
================

Utility functions for data processing.
"""
import numpy as np


def get_smoothing_kernel(
        fwhm: float,
        grid_resolution: float
) -> np.ndarray:
    """
    Calculate Gaussian smoothing kernel with a given full width at half
    maximum (FWHM).

    Args:
        fwhm: The full width at half maximum of the kernel.
        grid_resolution: The resolution of the underlying grid.

    Return:
        A numpy.ndarray containing the convolution kernel.
    """
    fwhm_n = fwhm / grid_resolution

    # Make sure width is uneven.
    width = int(fwhm_n * 3)
    x = np.arange(-(width // 2), width // 2 + 0.1)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2)
    k = np.exp(np.log(0.5) * (2.0 * r / fwhm_n) ** 2)
    k /= k.sum()

    return k
