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
    resample_data,
    calculate_footprint_weights,
    calculate_footprint_averages,
    interp_along_swath
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

    tbs = preprocessor_data.tbs_mw_gprof.data
    tbs_r = preprocessor_data_r.tbs_mw_gprof.data

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
        grids.GLOBAL.grid,
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
       grids.GLOBAL.grid,
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


def test_calculate_footprint_weights():
    """
    Test calculation of footprint weights for GMI and ensure that:

    - For a 1 degree beam width power 9km from center is reduced roughly to 0.5
    """
    spacecraft_position = (-22.633354, 28.659, 404286.7431640625)
    center = (-20.182945, 32.425606)
    lons = np.array([-20.182943, -20.139821750014576])
    lats = np.array([32.425606, 32.49671201563593])
    beam_width = 0.98
    weights = calculate_footprint_weights(
        lons,
        lats,
        center,
        spacecraft_position,
        beam_width
    )
    assert np.isclose(weights[1] / weights[0], 0.5, rtol=0.05)


def test_calculate_footprint_averages():
    """
    Test calculation of footprint for a discrete 2D dirac delta function and ensure that
    the antenna pattern is recovered in the output.
    """

    # Data to resample
    lons = np.linspace(-5, 5, 201)
    lats = np.linspace(-5, 5, 201)
    data = np.zeros((201, 201, 2))
    data[100, 100, :] = 1.0
    data = xr.DataArray(
        data,
        dims=("latitude", "longitude", "channels"),
        coords={
        "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
        }
    )
    data = xr.Dataset({
        "data": data
    })

    n_scans = 21
    n_pixels = 21
    lons = np.linspace(-0.05, 0.05, n_pixels)
    lats = np.linspace(-0.05, 0.05, n_scans)
    lons, lats = np.meshgrid(lons, lats)

    longitudes = xr.DataArray(
        lons,
        dims=("scans", "pixels")
    )
    latitudes = xr.DataArray(
        lats,
        dims=("scans", "pixels")
    )
    sensor_longitudes = xr.DataArray(
        lons[:, n_scans // 2],
        dims=("scans",)
    )
    sensor_latitudes = xr.DataArray(
        lats[:, n_scans // 2],
        dims=("scans",)
    )
    sensor_altitudes = xr.DataArray(
        160e3 * np.ones(n_scans),
        dims=("scans",)
    )

    results = calculate_footprint_averages(
        data,
        longitudes,
        latitudes,
        sensor_longitudes,
        sensor_latitudes,
        sensor_altitudes,
        2.0,
        area_of_influence=2.0
    )

    assert np.isclose(results["data"][10, 5], 0.5, atol=0.01).all()
    assert np.isclose(results["data"][10, 10], 1.0, atol=0.01).all()
    assert np.isclose(results["data"][10, 15], 0.5, atol=0.01).all()
    assert np.isclose(results["data"][5, 10], 0.5, atol=0.01).all()
    assert np.isclose(results["data"][15, 10], 0.5, atol=0.01).all()


def test_interp_along_swath():
    """
    Test interpolation of gridded data along a variable time field.
    """

    time = np.arange(
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-01T04:00:00"),
        np.timedelta64(1, "h")
    )

    field = np.tile(time[..., None, None], (1, 4, 4))
    scan_time = np.tile(time[..., None], (1, 4))

    dataset = xr.Dataset({
        "time": (("time",), time),
        "field": (("time", "latitude", "longitude"), field)
    })

    dataset_r = interp_along_swath(dataset, scan_time)
    assert (dataset_r["field"].data[..., 0].astype(time.dtype) == time).all()

    # Test out-of-bounds interpolation
    scan_time[:] = np.datetime64("2019-12-31T00:00:00")
    dataset_r = interp_along_swath(dataset, scan_time)
    assert (dataset_r["field"].data[..., 0].astype(time.dtype) == time[0]).all()

    scan_time[:] = np.datetime64("2020-01-31T00:00:00")
    dataset_r = interp_along_swath(dataset, scan_time)
    assert (dataset_r["field"].data[..., 0].astype(time.dtype) == time[-1]).all()
