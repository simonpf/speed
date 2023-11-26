"""
Tests for the speed.data.utils module.
"""
import numpy as np
from speed import grids
from speed.data.utils import (
    calculate_grid_resample_indices,
    calculate_swath_resample_indices,
    extract_rect,
    extract_scans,
    get_smoothing_kernel,
    resample_data
)
from gprof_nn.data.l1c import L1CFile

import xarray as xr


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


def test_extract_rect():
    """
    Test extraction of a retangular sub-domain of a dataset and ensure
    - that consecutive extraction keep 'lower_left_row' and 'lower_left_col'
      attibutes consistent w.r.t. to original grid.
    """
    lons, lats = grids.GLOBAL.grid.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]
    data = np.zeros((lats.size, lons.size))
    dataset = xr.Dataset({
        "latitude": ("latitude", lats),
        "longitude": ("longitude", lons),
        "data": (("latitude", "longitudes"), data)
    })

    sub_1 = extract_rect(dataset, 100, -100, 100, -100)
    assert sub_1.latitude.size == dataset.latitude.size - 200
    assert sub_1.longitude.size == dataset.longitude.size - 200
    assert sub_1.attrs["lower_left_row"] == 100
    assert sub_1.attrs["lower_left_col"] == 100

    sub_2 = extract_rect(sub_1, 100, -100, 100, -100)
    assert sub_2.latitude.size == dataset.latitude.size - 400
    assert sub_2.longitude.size == dataset.longitude.size - 400
    assert sub_2.attrs["lower_left_row"] == 200
    assert sub_2.attrs["lower_left_col"] == 200


def test_extract_scans(tmp_path, gmi_granule):
   """
   Extract scans corresponding to a given granule from a GMI file and ensure
   that:
       - Data loaded from the extract L1C file or from the ganule is the same.
   """
   path = extract_scans(gmi_granule, tmp_path)
   l1c_data_e = L1CFile(path).to_xarray_dataset()
   l1c_data_g = gmi_granule.open()

   lons_e = l1c_data_e.longitude.data
   lons_g = l1c_data_g.longitude_s1.data
   assert np.all(np.isclose(lons_e, lons_g))


def test_resample_data(preprocessor_data):
    """
    Test resample of gmi observations to global grid and ensure that:
      - Means of valid resample data is approximately conserved.


    """
    preprocessor_data_r = resample_data(preprocessor_data, grids.GLOBAL.grid, 10e3)

    tbs = preprocessor_data.tbs_mw.data
    tbs_r = preprocessor_data_r.tbs_mw.data

    valid = np.isfinite(tbs_r).any(-1)

    assert np.isclose(
        np.nanmean(tbs, axis=(0, 1)),
        tbs_r[valid].mean(axis=0),
        atol=1.0
    ).all()


def test_calculate_grid_resample_indices(gmi_granule):
    """
    Calculates indices for resampling global grid data to swath data and
    ensures that the differenes between swath coordinates and grid coordinates
    are less than the resolution of the global grid.
    """
    l1c_data = gmi_granule.open().rename({
        "latitude_s1": "latitude",
        "longitude_s1": "longitude"
    })
    indices = calculate_grid_resample_indices(
        l1c_data,
        grids.GLOBAL,
    )

    lons, lats = grids.GLOBAL.grid.get_lonlats()
    lons_r = lons[
        indices.row_index.data.ravel(),
        indices.col_index.data.ravel()
    ]
    lons_r = lons_r.reshape(l1c_data.longitude.data.shape)
    assert np.max(np.abs(lons_r - l1c_data.longitude.data)) < 0.036

    lats_r = lats[
        indices.row_index.data.ravel(),
        indices.col_index.data.ravel()
    ]
    lats_r = lats_r.reshape(l1c_data.latitude.data.shape)
    assert np.max(np.abs(lats_r - l1c_data.latitude.data)) < 0.036


def test_calculate_swath_resample_indices(gmi_granule):
   """
   Calculates indices for resampling global grid data to a swath and
   ensures that the differenes between swath coordinates and grid
   coordinates are less than the resolution of the global grid.
   """
   l1c_data = gmi_granule.open().rename({
       "latitude_s1": "latitude",
       "longitude_s1": "longitude"
   })
   indices = calculate_swath_resample_indices(
       l1c_data,
       grids.GLOBAL,
       100
   )

   lons_g, lats_g = grids.GLOBAL.grid.get_lonlats()
   lons = l1c_data.longitude.data
   lats = l1c_data.latitude.data

   lons_r = lons[
       indices.scan_index.data.ravel(),
       indices.pixel_index.data.ravel()
   ]
   lons_r = lons_r.reshape(lons_g.shape)
   valid = indices.scan_index.data >= 0
   assert np.max(np.abs(lons_r[valid] - lons_g[valid])) < 0.036

   lats_r = lats[
       indices.scan_index.data.ravel(),
       indices.pixel_index.data.ravel()
   ]
   lats_r = lats_r.reshape(lats_g.shape)
   valid = indices.scan_index.data >= 0
   assert np.max(np.abs(lats_r[valid] - lats_g[valid])) < 0.036
