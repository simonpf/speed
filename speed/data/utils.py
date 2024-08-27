"""
speed.data.utils
================

Utility functions for data processing.
"""
from copy import copy
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, Tuple

from gprof_nn.data.l1c import L1CFile
from pansat.time import to_datetime

import numpy as np
from pansat import Granule
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr


LOGGER = logging.getLogger(__name__)


def get_smoothing_kernel(fwhm: float, grid_resolution: float) -> np.ndarray:
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
    r = np.sqrt(x**2 + y**2)
    k = np.exp(np.log(0.5) * (2.0 * r / fwhm_n) ** 2)
    k /= k.sum()

    return k


def round_time(time: np.datetime64, step: np.timedelta64) -> np.datetime64:
    """
    Round time to given time step.

    Args:
        time: A numpy.datetime64 object representing the time to round.
        step: A numpy.timedelta64 object representing the time step to
            which to round the results.
    """
    if isinstance(time, datetime):
        time = to_datetime64(time)
    if isinstance(step, timedelta):
        step = to_timedelta64(step)
    time = time.astype("datetime64[s]")
    step = step.astype("timedelta64[s]")
    rounded = (
        np.datetime64(0, "s")
        + time.astype(np.int64) // step.astype(np.int64) * step
    )
    return rounded


def extract_rect(
    dataset: xr.Dataset, col_start: int, col_end: int, row_start: int, row_end: int
) -> xr.Dataset:
    """
    Extract a rectangular subset of a given dataset.

    Restricts the data in 'dataset' to the longitude and latitude ranges
    defined by 'col_start', 'col_end', 'row_start', 'row_end'. The indices
    of the lower left point of the rectangle are stored in attributes
    'lower_left_row' and 'lower_left_col' of the returned dataset.
    If these attributes are already present in 'dataset', the updated
    'lower_left_row' and 'lower_left_col' are updated by adding the
    row and column starts values to the existing values.

    Args:
        col_start: Start-column defining the rectangular region to extract.
        col_end: End-column defining the rectangular region to extract.
        row_start: Start-row defining the rectangular region to extract.
        row_end: End-row of the rectangular region to extract.

    Return:
        A new dataset pointing to a subset of 'dataset' with attributes
        'lower_left_row' and 'lower_left_col' identifying the lower-left
        corner of the extracted domain.
    """
    dataset_new = dataset[
        {"longitude": slice(col_start, col_end), "latitude": slice(row_start, row_end)}
    ]
    if "lower_left_row" in dataset_new.attrs:
        dataset_new.attrs["lower_left_row"] += row_start
        dataset_new.attrs["lower_left_col"] += col_start
    else:
        dataset_new.attrs["lower_left_row"] = row_start
        dataset_new.attrs["lower_left_col"] = col_start
    return dataset_new


def extract_scans(granule: Granule, dest: Path, min_scans: Optional[int] = None) -> Path:
    """
    Extract and write scans from L1C file into a separate file.

    Args:
        granule: A pansat granule specifying a subset of an orbit.
        dest: A directory to which the extracted scans will be written.
        min_scans: A minimum number of scans to extract.

    Return:
        The path of the file containing the extracted scans.
    """
    scan_start, scan_end = granule.primary_index_range
    n_scans = scan_end - scan_start
    if min_scans is not None and n_scans < min_scans:
        scan_c = (scan_end + scan_start) // 2
        scan_start = scan_c - min_scans // 2
        scan_end = scan_start + min_scans
    l1c_path = granule.file_record.local_path
    l1c_file = L1CFile(granule.file_record.local_path)
    output_filename = dest / l1c_path.name
    l1c_file.extract_scan_range(scan_start, scan_end, output_filename)
    return output_filename


def get_useful_scan_range(
        data: xr.Dataset,
        reference_variable: str,
        min_scans: int = 256,
        margin: int = 64
):
    """
    Determine scan range containing valid reference data.

    Args:
        data: A xarray.Dataset containing satellite observations in their on_swath sampling.
        reference_variable: The variable to use to determine scans with valid reference
        min_scans: The minimum name of scans in the returned range.
        margin: The number of scans by which to expand the valid scan range.

    """
    ref_data = data[reference_variable].data
    valid_scans = np.where(np.any(np.isfinite(ref_data), -1))[0]
    scan_start = max(valid_scans[0] - margin, 0)
    scan_end = min(valid_scans[-1] + margin, data.scan.size)
    n_scans = scan_end - scan_start
    if n_scans < min_scans:
        scan_c = int(0.5 * (scan_end + scan_start))
        scan_start = max(scan_c - min_scans // 2, 0)
        scan_end = min(scan_start + min_scans, ref_data.shape[0])
    return scan_start, scan_end


def save_data_on_swath(
    sensor_name: str,
    reference_data_name: str,
    preprocessor_data: xr.Dataset,
    reference_data: xr.Dataset,
    output_path: Path,
    min_scans: int = 96,
) -> None:
    """
    Save collocations in on_swath format.

    Args:
        sensor_name: The name of the sensor that the input data stems from.
        reference_data_name: The name of reference data source.
        preprocessor_data: A xarray.Dataset containing the retrieval input data
            from the preprocessor.
        reference_data: A xarray.Dataset containing the reference data.
        output_path: The base folder to which to write the collocation files.
        min_scans: Constrain the size of the saved collocation to at least
            this number of scans.

    """
    output_path = Path(output_path)
    on_swath_path = output_path / "on_swath"
    on_swath_path.mkdir(exist_ok=True, parents=True)

    time = to_datetime(preprocessor_data.scan_time.mean().data)
    fname = time.strftime(f"{reference_data_name}_{sensor_name}_%Y%m%d%H%M%S.nc")

    output_file = on_swath_path / fname
    preprocessor_data.to_netcdf(output_file, group="input_data")
    reference_data.to_netcdf(output_file, group="reference_data", mode="a")
    return time


def save_data_gridded(
    sensor_name: str,
    reference_data_name: str,
    time: np.datetime64,
    preprocessor_data: xr.Dataset,
    reference_data: xr.Dataset,
    output_path: Path,
    min_size: int = 96,
) -> None:
    """
    Save collocations in gridded format.

    Args:
        sensor_name: The name of the sensor that the input data stems from.
        reference_data_name: The name of reference data source.
        time: The mean scan-time used to save the collocation in on_swath
            format.
        preprocessor_data: A xarray.Dataset containing the retrieval input data
            from the preprocessor.
        reference_data: A xarray.Dataset containing the reference data.
        output_path: The base folder to which to write the collocation files.

    """
    if (reference_data.latitude.size != preprocessor_data.latitude.size) or (
        reference_data.longitude.size != preprocessor_data.longitude.size
    ):
        row_start = reference_data.attrs.get("lower_left_row", 0)
        n_rows = reference_data.latitude.size
        n_cols = reference_data.longitude.size
        col_start = reference_data.attrs.get("lower_left_col", 0)
        preprocessor_data = extract_rect(
            preprocessor_data,
            col_start,
            col_start + n_cols,
            row_start,
            row_start + n_rows,
        )

    output_path = Path(output_path)
    gridded_path = output_path / "gridded"
    gridded_path.mkdir(exist_ok=True, parents=True)

    if reference_data.longitude[0] < -179 and reference_data.longitude[-1] > 179:

        surface_precip = reference_data.surface_precip.data
        obs = preprocessor_data.observations.data
        valid = np.isfinite(surface_precip) * np.isfinite(obs).any(-1)
        row_inds, col_inds = np.where(valid)

        # If collocation crosses the date line, we simply shift it to the left
        # so that all valid data is right of the meridian line.j
        if np.any(valid[:, 0]) and np.any(valid[:, -1]):
            n_cols = valid.shape[1]
            # Find last column with valid indices.
            n_roll = np.where(valid[:, : n_cols // 2].any(0))[0][-1] + 1
            preprocessor_data = preprocessor_data.roll(
                longitude=-n_roll,
                roll_coords=True
            )
            reference_data = reference_data.roll(
                longitude=-n_roll,
                roll_coords=True
            )

    surface_precip = reference_data.surface_precip.data
    obs = preprocessor_data.observations.data
    valid = np.isfinite(surface_precip) * np.isfinite(obs).any(-1)
    row_inds, col_inds = np.where(valid)

    # Try to expand collocation by 32 pixels in each direction.
    row_start = max(row_inds.min() - 32, 0)
    row_end = min(row_inds.max() + 32, valid.shape[0])
    col_start = max(col_inds.min() - 32, 0)
    col_end = min(col_inds.max() + 32, valid.shape[1])

    n_rows = row_end - row_start
    if n_rows < min_size:
        row_c = int(0.5 * (row_start + row_end))
        row_start = max(row_c - min_size // 2, 0)
        row_end = min(row_start + min_size, valid.shape[0])

    n_cols = col_end - col_start
    if n_cols < min_size:
        col_c = int(0.5 * (col_start + col_end))
        col_start = max(col_c - min_size // 2, 0)
        col_end = min(col_start + min_size, valid.shape[1])

    preprocessor_data = extract_rect(
        preprocessor_data, col_start, col_end, row_start, row_end
    )
    reference_data = extract_rect(
        reference_data, col_start, col_end, row_start, row_end
    )

    time = to_datetime(time)
    fname = time.strftime(f"{reference_data_name}_{sensor_name}_%Y%m%d%H%M%S.nc")

    output_file = gridded_path / fname

    encoding = {var: {"compression": "zstd"} for var in preprocessor_data}
    preprocessor_data.to_netcdf(output_file, group="input_data", encoding=encoding)
    encoding = {var: {"compression": "zstd"} for var in reference_data}
    reference_data.to_netcdf(output_file, group="reference_data", mode="a", encoding=encoding)


def resample_data(
        dataset,
        target_grid,
        radius_of_influence=5e3,
        new_dims=("latitude", "longitude")
) -> xr.Dataset:
    """
    Resample xarray.Dataset data to global grid.

    Args:
        dataset: xr.Dataset containing data to resample to global grid.
        target_grid: A pyresample.AreaDefinition defining the global grid
            to which to resample the data.

    Return:
        An xarray.Dataset containing the give dataset resampled to
        the global grid.
    """
    lons = dataset.longitude.data
    lats = dataset.latitude.data

    if isinstance(target_grid, tuple):
        lons_t, lats_t = target_grid
        shape = lons_t.shape
    else:
        lons_t, lats_t = target_grid.get_lonlats()
        shape = target_grid.shape

    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )

    if lons.shape != lats.shape:
        lons, lats = np.meshgrid(lons, lats)

    swath = SwathDefinition(lons=lons, lats=lats)
    target = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    resampled = {}
    resampled["latitude"] = (("latitude",), lats_t[:, 0])
    resampled["longitude"] = (("longitude",), lons_t[0, :])

    for var in dataset:
        if var in ["latitude", "longitude"]:
            continue
        data = dataset[var].data
        if data.ndim == 1 and lons.ndim > 1:
            data = np.broadcast_to(data[:, None], lons.shape)

        dtype = data.dtype
        if np.issubdtype(dtype, np.datetime64):
            fill_value = np.datetime64("NaT")
        elif np.issubdtype(dtype, np.integer):
            fill_value = -9999
        elif dtype == np.int8:
            fill_value = -1
        else:
            fill_value = np.nan


        data_r = kd_tree.get_sample_from_neighbour_info(
            "nn", target.shape, data, ind_in, ind_out, inds, fill_value=fill_value
        )

        data_full = np.zeros(shape + data.shape[lons.ndim:], dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            data_full = np.nan * data_full
        elif np.issubdtype(dtype, np.datetime64):
            data_full[:] = np.datetime64("NaT")
        elif dtype == np.int8:
            data_full[:] = -1
        else:
            data_full[:] = -9999

        data_full[valid_pixels] = data_r
        resampled[var] = (new_dims + dataset[var].dims[lons.ndim:], data_full)

    return xr.Dataset(resampled)


def calculate_grid_resample_indices(dataset, target_grid):
    """
    Calculate scan and pixel indices of closest pixels in target grid.

    Args:
        dataset: An xarray.Dataset containing satellite data on a swath.
        target_grid: A grid to which to map to the swath data.

    Return:
        An xarray.Dataset containing variables 'row_index' and
        'pixel_index' that can be used to map swath data to the
        global grid.
    """
    lons = dataset.longitude.data
    lats = dataset.latitude.data
    lons_t, lats_t = target_grid.get_lonlats()
    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )
    row_inds, col_inds = np.where(valid_pixels)

    swath = SwathDefinition(lons=lons, lats=lats)
    source = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    info = kd_tree.get_neighbour_info(
        source, swath, radius_of_influence=5e3, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    dims = ("scan", "pixel")
    resampled = {}

    row_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", swath.shape, row_inds, ind_in, ind_out, inds, fill_value=-1
    )
    col_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", swath.shape, col_inds, ind_in, ind_out, inds, fill_value=-1
    )

    return xr.Dataset(
        {"row_index": (dims, row_inds_r), "col_index": (dims, col_inds_r)}
    )


def calculate_swath_resample_indices(dataset, target_grid, radius_of_influence):
    """
    Calculate scan and pixel indices of closest pixels in target grid.

    Args:
        dataset: An xarray.Dataset containing satellite data on a swath.
        target_grid: A grid to which to map to the swath data.

    Return:
        An xarray.Dataset containing variables 'scan_indices' and
        'pixel_indices' that map each point in the global grid to
        the closest swath pixel. If the grid point is not withing the
        given radius of influence, the index is set to -1.
    """
    lons = dataset.longitude.data
    lats = dataset.latitude.data
    lons_t, lats_t = target_grid.get_lonlats()
    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )

    swath = SwathDefinition(lons=lons, lats=lats)
    target = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    pixel_inds, scan_inds = np.meshgrid(dataset.pixel.data, dataset.scan.data)

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    dims = ("latitude", "longitude")
    resampled = {}

    scan_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", target.shape, scan_inds, ind_in, ind_out, inds, fill_value=-1
    )
    scan_inds_gridded = -np.ones(target_grid.shape, dtype=np.int16)
    scan_inds_gridded[valid_pixels] = scan_inds_r

    pixel_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", target.shape, pixel_inds, ind_in, ind_out, inds, fill_value=-1
    )
    pixel_inds_gridded = -np.ones(target_grid.shape, dtype=np.int16)
    pixel_inds_gridded[valid_pixels] = pixel_inds_r

    return xr.Dataset(
        {
            "scan_index": (dims, scan_inds_gridded),
            "pixel_index": (dims, pixel_inds_gridded),
        }
    )


ANCILLARY_VARIABLES = [
    "wet_bulb_temperature",
    "two_meter_temperature",
    "lapse_rate",
    "total_column_water_vapor",
    "surface_temperature",
    "moisture_convergence",
    "leaf_area_index",
    "snow_depth",
    "orographic_wind",
    "10m_wind",
    "surface_type",
    "mountain_type",
    "land_fraction",
    "ice_fraction",
    "quality_flag",
    "sunglint_angle",
    "airlifting_index"
]

def save_ancillary_data(
        input_data: xr.Dataset,
        time: datetime,
        path: Path
) -> None:
    """
    Save ancillary data in separate folder.

    Args:
        input_data: A xarray.Dataset containing all retrieval input data
            for a single training scene.
        time: The median time of the scene.
        path: The base folder in which to store the extracted training
            scenes.
    """
    date_str = time.strftime("%Y%m%d%H%M%S")
    filename = f"ancillary_{date_str}.nc"

    output_path = path / "ancillary"
    output_path.mkdir(exist_ok=True, parents=True)

    anc_vars = ANCILLARY_VARIABLES
    if "latitude" not in input_data.dims:
        anc_vars = copy(anc_vars) + ["latitude", "longitude"]

    ancillary_data = input_data[anc_vars]
    ancillary_data.to_netcdf(output_path / filename)


def save_input_data(
        sensor: str,
        input_data: xr.Dataset,
        time: datetime,
        path: Path
) -> None:
    """
    Save input data in separate folder.

    Args:
        input_data: A xarray.Dataset containing all retrieval input data
            for a single training scene.
        time: The median time of the scene.
        path: The base folder in which to store the extracted training
            scenes.
    """
    date_str = time.strftime("%Y%m%d%H%M%S")
    filename = f"{sensor}_{date_str}.nc"

    output_path = path / f"{sensor}"
    output_path.mkdir(exist_ok=True, parents=True)

    uint16_max = 2 ** 16 - 1
    int16_min = 2 ** 15
    encoding = {
        "observations": {"dtype": "uint16", "_FillValue": uint16_max, "scale_factor": 0.01, "compression": "zstd"},
        "earth_incidence_angle": {"dtype": "int16", "_FillValue": -(2e-15), "scale_factor": 0.01, "compression": "zstd"},
    }
    vars = [
        var for var in ["observations", "earth_incidence_angle"]
        if var in input_data
    ]
    encoding = {var: encoding[var] for var in vars}
    input_data = input_data[vars]
    input_data.to_netcdf(output_path / filename, encoding=encoding)


TARGET_VARIABLES = [
    "surface_precip",
    #"surface_precip_fpavg",
    "radar_quality_index",
    "gauge_correction_factor",
    "valid_fraction",
    "precip_fraction",
    "snow_fraction",
    "hail_fraction",
    "convective_fraction",
    "stratiform_fraction",
    "time",
    "precipitation_type",
    "rain_water_content",
    "total_water_content",
    "rain_water_path",
    "snow_water_path",
    "latent_heating"
]


def save_target_data(
        reference_data: xr.Dataset,
        time: datetime,
        path: Path,
        include_swath_coords: bool = False
) -> None:
    """
    Save input data in separate folder.

    Args:
        reference_data: A xarray.Dataset containing all retrieval reference data
            for a single training scene.
        time: The median time of the scene.
        path: The base folder in which to store the extracted training
            scenes.
        include_swath_coords: If True will include the swath coordinates of the input pixels
            on the on_swath sensor geometry allowing results to be efficiently
            remapped to the gridded data.
    """
    date_str = time.strftime("%Y%m%d%H%M%S")
    filename = f"target_{date_str}.nc"

    output_path = path / "target"
    output_path.mkdir(exist_ok=True, parents=True)

    target_variables = [
        var for var in TARGET_VARIABLES if var in reference_data
    ]
    if include_swath_coords:
        target_variables += [
            "scan_index",
            "pixel_index"
        ]

    uint16_max = 2 ** 16 - 1
    encoding = {
        "surface_precip": {"dtype": "uint16", "_FillValue": uint16_max, "scale_factor": 0.01, "compression": "zstd"},
        "radar_quality_index": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "valid_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "precip_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "snow_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "hail_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "convective_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "stratiform_fraction": {"dtype": "uint8", "_FillValue": 255, "scale_factor": 1.0/254.0, "compression": "zstd"},
        "precipitation_type": {"dtype": "int8", "compression": "zstd"},
        "total_water_content": {"dtype": "float32", "compression": "zstd"},
        "rain_water_content": {"dtype": "float32", "compression": "zstd"},
        "total_water_content": {"dtype": "float32", "compression": "zstd"},
        "rain_water_path": {"dtype": "float32", "compression": "zstd"},
        "snow_water_path": {"dtype": "float32", "compression": "zstd"},
        "latent_heating": {"dtype": "float32"}
    }

    target_data = reference_data[target_variables]
    encoding = {var: encoding[var] for var in target_variables if var in encoding}
    target_data.to_netcdf(output_path / filename, encoding=encoding)


def save_geo_data(
        geo_data: xr.Dataset,
        time: datetime,
        path: Path
) -> None:
    """
    Save GEO data in separate folder.

    Args:
        geo_data: A xarray.Dataset containing all retrieval input data
            for a single training scene.
        time: The median time of the scene.
        path: The base folder in which to store the extracted training
            scenes.
    """
    date_str = time.strftime("%Y%m%d%H%M%S")
    filename = f"geo_{date_str}.nc"
    uint16_max = 2 ** 16 - 1
    encoding = {
        "observations": {"dtype": "uint16", "_FillValue": uint16_max, "scale_factor": 0.01, "compression": "zstd"},
    }
    output_path = path / "geo"
    output_path.mkdir(exist_ok=True, parents=True)
    geo_data.to_netcdf(output_path / filename, encoding=encoding)


def save_geo_ir_data(
        geo_ir_data: xr.Dataset,
        time: datetime,
        path: Path
) -> None:
    """
    Save GEO IR data in separate folder.

    Args:
        geo_ir_data: A xarray.Dataset containing all retrieval input data
            for a single training scene.
        time: The median time of the scene.
        path: The base folder in which to store the extracted training
            scenes.
    """
    date_str = time.strftime("%Y%m%d%H%M%S")
    filename = f"geo_ir_{date_str}.nc"
    uint16_max = 2 ** 16 - 1
    encoding = {
        "observations": {"dtype": "uint16", "_FillValue": uint16_max, "scale_factor": 0.01, "compression": "zstd"},
    }
    output_path = path / "geo_ir"
    output_path.mkdir(exist_ok=True, parents=True)
    geo_ir_data = geo_ir_data.rename(tbs_ir="observations")
    geo_ir_data.to_netcdf(output_path / filename, encoding=encoding)


def extract_scenes(
        collocation_file: Path,
        output_folder: Path,
        size: int = 256,
        overlap: float = 0.0,
        min_valid_input_frac: float = 0.75,
        min_valid_ref_frac: float = 0.25,
        include_geo: bool = False,
        include_geo_ir: bool = False
) -> None:
    """
    Extract scenes of a given size from a collocation file suitable for training spatially-resolved
    retrievals. Each scene is stored in a different files and files are  split up by input source,
    ancillary data and target files.

    Args:
        collocation_file: A path pointing to the file containing the collocation data from which
            to extract training scenes.
        output_folder: The folder to which to write the extracted scenes.
        size: The size of the scenes.
        overlap: The maximum overlap in any direction between two scenes.
        min_valid_input_frac: The minimum fraction of pixels with valid observations for a scene
            to be considered valid.
        min_valid_ref_frac: The minimum fraction of pixels with valid reference data for a scene
            to be considered valid.
        include_geo: If True, will try to extract GEO data for each training scene.
        include_geo_ir: If True, will try to extract GEO IR data for each training scene.
    """
    sensor_name = collocation_file.name.split("_")[1]
    input_data = xr.load_dataset(collocation_file, group="input_data")
    reference_data = xr.load_dataset(collocation_file, group="reference_data")
    if include_geo:
        geo_data = xr.load_dataset(collocation_file, group="geo")
        if "scans" in geo_data.dims:
            geo_data = geo_data.rename(scans="scan", pixels="pixel")
    else:
        geo_data = None

    if include_geo_ir:
        geo_ir_data = xr.load_dataset(collocation_file, group="geo_ir")
        if "scans" in geo_ir_data.dims:
            geo_ir_data = geo_ir_data.rename(scans="scan", pixels="pixel")
    else:
        geo_ir_data = None

    spatial_dims = input_data.observations.dims[:2]
    n_rows = input_data[spatial_dims[0]].size
    n_cols = input_data[spatial_dims[1]].size

    valid_input = np.any(np.isfinite(input_data.observations.data), -1)
    valid_output = np.isfinite(reference_data.surface_precip.data)
    valid = valid_input * valid_output

    row_inds, col_inds = np.where(valid)
    within = (
        (row_inds >= size // 2) * (row_inds < n_rows - size // 2) *
        (col_inds >= size // 2) * (col_inds < n_cols - size // 2)
    )
    row_inds = row_inds[within]
    col_inds = col_inds[within]

    max_retries = 200
    curr_try = 0
    n_inds = len(row_inds)


    while curr_try < max_retries and len(row_inds) > 0:

        ind = np.random.randint(len(row_inds))
        c_row = row_inds[ind]
        c_col = col_inds[ind]

        row_start = c_row - size // 2
        row_end = c_row + size // 2
        col_start = c_col - size // 2
        col_end = c_col + size // 2
        slices = [slice(row_start, row_end), slice(col_start, col_end)]

        inpt = input_data[{name: slc for name, slc in zip(spatial_dims, slices)}]
        ref = reference_data[{name: slc for name, slc in zip(spatial_dims, slices)}]

        valid_input_frac = valid_input[slices[0], slices[1]].mean()
        if valid_input_frac < min_valid_input_frac:
            row_inds = np.concatenate([row_inds[:ind], row_inds[ind + 1:]])
            col_inds = np.concatenate([col_inds[:ind], col_inds[ind + 1:]])
            continue
        valid_ref_frac = valid_input[slices[0], slices[1]].mean()
        if valid_ref_frac < min_valid_ref_frac:
            row_inds = np.concatenate([row_inds[:ind], row_inds[ind + 1:]])
            col_inds = np.concatenate([col_inds[:ind], col_inds[ind + 1:]])
            continue

        if "lower_lef_col" in inpt.attrs:
            inpt.attrs["lower_left_col"] += col_start
            inpt.attrs["lower_left_row"] += row_start
            ref.attrs["lower_left_col"] += col_start
            ref.attrs["lower_left_row"] += row_start
        if "scan_start" in inpt.attrs:
            inpt.attrs["scan_start"] += row_start
            inpt.attrs["scan_end"] = inpt.attrs["scan_start"] + row_end
            inpt.attrs["pixel_start"] = col_start
            inpt.attrs["pixel_end"] = col_end

        scan_times = inpt.scan_time.data
        scan_times = scan_times[np.isfinite(scan_times)]
        time = scan_times[0] + np.median(scan_times - scan_times[0])
        time = to_datetime(time)

        #save_ancillary_data(inpt, time, output_folder)
        save_input_data(sensor_name, inpt, time, output_folder)
        save_target_data(ref, time, output_folder)
        if geo_data is not None:
            save_geo_data(
                geo_data[{name: slc for name, slc in zip(spatial_dims, slices)}],
                time,
                output_folder
            )
        if geo_ir_data is not None:
            save_geo_ir_data(
                geo_ir_data[{name: slc for name, slc in zip(spatial_dims, slices)}],
                time,
                output_folder
            )

        margin = max((1.0 - overlap) * size, 1)
        covered = (
            (row_inds >= c_row - margin) * (row_inds <= c_row + margin) *
            (col_inds >= c_col - margin) * (col_inds <= c_col + margin)
        )
        n_inds = len(row_inds)
        row_inds = row_inds[~covered]
        col_inds = col_inds[~covered]
        if n_inds == len(row_inds):
            curr_try += 1
        else:
            curr_try = 0


def extract_evaluation_data(
        collocation_file_gridded: Path,
        collocation_file_on_swath: Path,
        output_folder: Path,
        include_geo: bool = False,
        include_geo_ir: bool = False,
        min_size: int = 256
) -> None:
    """
    This function extract full collocation data from collocation file and stores it in the same
    way 'extract_scenes' but keeps collocations in both gridded and on_swath format.

    Args:
        collocation_file_gridded: A path pointing to a file containing a SPEED collocation
            in on_swath sampling.
        collocation_file_gridded: A path pointing to a file containing a SPEED collocation
            in regridded format.
        output_folder: The folder to which to write the extracted scenes.
        include_geo: If 'True', will extract GEO observations for all evaluation scenes.
        include_geo_ir: If 'True', will extract GEO IR observations for all evaluation scenes.
        min_size: Scene below that size will be discarded.
    """
    parts = collocation_file_on_swath.name.split("_")
    sensor_name = parts[-2]
    time_str = parts[-1][:-3]
    time = datetime.strptime(time_str, "%Y%m%d%H%M%S")


    with xr.load_dataset(collocation_file_gridded, group="input_data") as input_data:
        n_lats = input_data.sizes["latitude"]
        n_lons = input_data.sizes["longitude"]
    if n_lats < min_size or n_lons < min_size:
        LOGGER.info(
            "Skipping scene %s because it is smaller than 256 x 256 (%s x %s).",
            time_str, n_lats, n_lons
        )
        return None

    # Try and load geo and geo_ir data.
    if include_geo:
        geo_data = xr.load_dataset(collocation_file_on_swath, group="geo")
        save_geo_data(geo_data, time, output_folder / "on_swath")
    if include_geo_ir:
        geo_ir_data = xr.load_dataset(collocation_file_on_swath, group="geo_ir")
        save_geo_ir_data(geo_ir_data, time, output_folder / "on_swath")

    # Extract on_swath data.
    input_data = xr.load_dataset(collocation_file_on_swath, group="input_data")
    input_data.attrs["gpm_input_file"] = input_data.attrs.pop("gpm_input_file")
    reference_data = xr.load_dataset(collocation_file_on_swath, group="reference_data")
    save_ancillary_data(input_data, time, output_folder / "on_swath")
    save_input_data(sensor_name, input_data, time, output_folder / "on_swath")
    save_target_data(reference_data, time, output_folder / "on_swath")

    #surface_precip_fpavg = reference_data.surface_precip_fpavg
    reference_data = xr.load_dataset(collocation_file_gridded, group="reference_data")
    pixel_inds = reference_data.pixel_index
    scan_inds = reference_data.scan_index
    #surface_precip_fpavg = surface_precip_fpavg[{"scans": scan_inds, "pixels": pixel_inds}]
    #invalid = scan_inds.data < 0
    #surface_precip_fpavg.data[invalid] = np.nan
    #reference_data["surface_precip_fpavg"] = surface_precip_fpavg

    gpm_input_file = input_data.attrs["gpm_input_file"]

    # Extract gridded data.
    input_data = xr.load_dataset(collocation_file_gridded, group="input_data")
    input_data.attrs["gpm_input_file"] = gpm_input_file
    save_ancillary_data(input_data, time, output_folder / "gridded")
    save_input_data(sensor_name, input_data, time, output_folder / "gridded")
    save_target_data(
        reference_data,
        time,
        output_folder / "gridded",
        include_swath_coords = True
    )
    if include_geo:
        geo_data = xr.load_dataset(collocation_file_gridded, group="geo")
        save_geo_data(geo_data, time, output_folder / "gridded")
    if include_geo_ir:
        geo_ir_data = xr.load_dataset(collocation_file_gridded, group="geo_ir")
        save_geo_ir_data(geo_ir_data, time, output_folder / "gridded")


def extract_training_data(
        collocation_file: Path,
        include_geo: bool = False,
        include_geo_ir: bool = False
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Extract tabular training data from collocation.

    Args:
        collocation_file: A path object pointing to a collocation file from which to extract
             the training data.
        include_geo: If 'True' will try to load geo data and include it in the
             training data.
        include_geo_ir: If 'True' will try to load geo IR data and include it in the
             training data.

    Return:
        A tuple ``(input_data, ancillary_data, target_data)`` containing the input, ancillary
        and target data for the retrieval.
    """
    input_data = xr.load_dataset(collocation_file, group="input_data")
    reference_data = xr.load_dataset(collocation_file, group="reference_data")
    valid_output = np.isfinite(reference_data.surface_precip.data)

    if "scan" in input_data.dims:
        spatial_dims = ("scan", "pixel")
    else:
        spatial_dims = ("latitude", "longitude")

    valid_input = np.zeros_like(valid_output)
    valid_input += np.any(np.isfinite(input_data.observations.data), -1)

    if include_geo:
        geo = xr.load_dataset(collocation_file, group="geo").transpose("time", "channel", ...)
        delta_t = (reference_data.time - geo.time).transpose("time", ...)
        nearest_ind = np.argmin(np.abs(delta_t.data), axis=0)
        geo = geo.drop_vars("time")
        geo["nearest_ind"] = (spatial_dims, nearest_ind.astype(np.uint8))
    else:
        geo = None

    if include_geo_ir:
        geo_ir = xr.load_dataset(collocation_file, group="geo_ir")
        geo_ir = geo_ir.rename(tbs_ir="observations")
        delta_t = (reference_data.time - geo_ir.time).transpose("time", ...)
        nearest_ind = np.argmin(np.abs(delta_t.data), axis=0)
        geo_ir = geo_ir.drop_vars("time")
        geo_ir["nearest_ind"] = (spatial_dims, nearest_ind.astype(np.uint8))
    else:
        geo_ir = None

    valid = valid_input * valid_output

    # GEO data
    if geo is not None:
        geo_data = xr.Dataset({
            "observations": (
                ("samples", "time", "channel"), np.transpose(geo.observations.data[..., valid], (2, 0, 1))
            ),
            "nearest_time_step": (
                ("samples"), np.transpose(geo.nearest_ind.data[..., valid]).transpose()
            )
        })
    else:
        geo_data = None

    # GEO-IR data
    if geo_ir is not None:
        geo_ir_data = xr.Dataset({
            "observations": (("samples", "time"), geo_ir.observations.data[:, valid].transpose()),
            "nearest_ind": (("samples"), geo_ir.nearest_ind.data[valid].transpose())
        })
    else:
        geo_ir_data = None

    ancillary_data = xr.Dataset({
        name: (("samples",), input_data[name].data[valid])
        for name in ANCILLARY_VARIABLES
    })
    for name in ANCILLARY_VARIABLES:
        array = ancillary_data[name]
        if np.issubdtype(array.dtype, np.floating):
            invalid = array < -9990
            array.data[invalid] = np.nan
            ancillary_data[name] = array.astype(np.float32)

    lats = input_data["latitude"]
    lons = input_data["longitude"]
    if lons.ndim < 2:
        lats, _ = xr.broadcast(lats, reference_data["surface_precip"])
        lons, _ = xr.broadcast(lons, reference_data["surface_precip"])
        lats = lats.transpose("latitude", "longitude")
        lons = lons.transpose("latitude", "longitude")

    ancillary_data["latitude"] = (("samples",), lats.data[valid])
    ancillary_data["longitude"] = (("samples",), lons.data[valid])

    pmw_data = xr.Dataset({
        name: (("samples", "channel"), input_data[name].data[valid])
        for name in ["observations", "earth_incidence_angle"]
    })

    target_names = [
        "time",
        "surface_precip",
        "radar_quality_index",
        "valid_fraction",
        "precip_fraction",
        "snow_fraction",
        "hail_fraction",
        "convective_fraction",
        "stratiform_fraction",
    ]
    target_data = xr.Dataset({
        name: (("samples",), reference_data[name].data[valid])
        for name in target_names if name in reference_data
    })

    return pmw_data, geo_data, geo_ir_data, ancillary_data, target_data,


def lla_to_ecef(
        lon: np.ndarray,
        lat: np.ndarray,
        alt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert longitude, latitude and altidue to earth-centered earth-fixed (ECEF)
    coordinates.

    Args:
        lon: A numpy.ndarray containing longitude coordinates.
        lat: A numpy.ndarray containing latitude coordinates.
        alt: A numpy.ndarray containing altitude coordinates.

    Return:
        A tuple ``(x, y, z)`` containing the ECEF x-, y-, and z-coordinates, respectively
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    e_sq = 1 - (b**2 / a**2)  # Square of eccentricity

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate N, the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)

    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((b**2 / a**2) * N + alt) * np.sin(lat_rad)

    return x, y, z


def calculate_footprint_weights(
        lons: np.ndarray,
        lats: np.ndarray,
        center: Tuple[float, float],
        spacecraft_position: Tuple[float, float, float],
        beam_width: float
) -> np.ndarray:
    """
    Calculate footprint weights for pixels on surface.

    Args:
        lons: A numpy.ndarray containing the longitude coordinates of the points on the surface.
        lats: A numpy.ndarray containing the latitude coordinates of the points on the surface.
        lon_sc: Longitude coordinate of the spacecraft.
        lat_sc: Latitude coordinate of the spacecraft.
        alt_sc: The altitude of the spacecraft in meters.
        beam_width: The sensor beam width.

    Return:
        A weigth array of the same shape as 'lons' and 'lats' containing the antenna sensitivity
        assuming a Gaussian antenna pattern with the given beam width.
    """
    lon_sc, lat_sc, alt_sc = spacecraft_position
    x_sc, y_sc, z_sc = lla_to_ecef(lon_sc, lat_sc, alt_sc)
    vec_sc = np.stack((x_sc, y_sc, z_sc), -1)

    lon_c, lat_c = center
    x_c, y_c, z_c = lla_to_ecef(lon_c, lat_c, 0.0)
    vec_c = np.stack((x_c, y_c, z_c), -1) - vec_sc

    x, y, z = lla_to_ecef(lons, lats, 0)
    vec = np.stack((x, y, z), -1) - vec_sc

    vec_c = np.broadcast_to(vec_c, vec.shape)
    angs = np.rad2deg(
        np.arccos(
            np.minimum((vec * vec_c).sum(-1) / np.sqrt(np.sum(vec ** 2, -1) * np.sum(vec_c ** 2, -1)), 1.0)
        )
    )
    vec_c = np.stack((x_c, y_c, z_c), -1)[None, None] - vec_sc
    ap = np.exp(np.log(0.5) * (2.0 * angs / beam_width) ** 2)
    return ap


def calculate_footprint_averages(
        data: xr.DataArray,
        longitudes: xr.DataArray,
        latitudes: xr.DataArray,
        sensor_longitudes: xr.DataArray,
        sensor_latitudes: xr.DataArray,
        sensor_altitudes: xr.DataArray,
        beam_width: float,
        area_of_influence: float = 1.0,
        sensor_time: Optional[xr.DataArray] = None,
        max_time_diff: Optional[np.timedelta64] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a given data array by calculating footprint averages.

    Args:
        data: A xarray.DataArray containing the data to resample.
        longitudes: A xarray.DataAttary containing the longitude coordinates to which to resample
            'data'.
        latitudes: A xarray.DataArray containing the latitude coordinates to which to resample
            'data'.
        sensor_longitudes: An xarray.DataArray containing the longitude coordinates of the sensor position
            corresponding to all observed pixels.
        sensor_latitudes: An xarray.DataArray containing the latitude coordinates of the sensor position
            corresponding to all observed pixels.
        beam_width: The beam width of the sensor.
        area_of_influence: The extent of the are in degree to consider for the footprint averaging.
        sensor_time: A DataArray of the same shape as longitudes and latitudes containing the observations
            times for all pixels.
        max_time_diff: An optional limit on the time difference between the data to resample and the
             observations to limit the amount of calculations.

    Return:
        A tuple of number arrays ``(data_fpavg, valid_fractions)`` containing the footprint averaged
        data and the corresponding fraction of valid pixels withing the footprint.
    """
    sensor_longitudes, _ = xr.broadcast(sensor_longitudes, latitudes)
    sensor_latitudes, _ = xr.broadcast(sensor_latitudes, latitudes)
    sensor_altitudes, _ = xr.broadcast(sensor_altitudes, latitudes)

    sensor_dims = sensor_latitudes.dims[:2]
    spatial_dims = data.latitude.ndim
    results = {
        var: (
            (sensor_dims + data[var].dims[:2])[:2 + data[var].data.ndim - spatial_dims],
            np.nan * np.zeros(latitudes.shape + data[var].shape[2:], dtype=data[var].dtype)
        ) for var in data if var != "time"
    }

    lons_data = data.longitude.data
    lats_data = data.latitude.data
    lon_min, lon_max = lons_data.min(), lons_data.max()
    lat_min, lat_max = lats_data.min(), lats_data.max()

    lons = longitudes.data
    lats = latitudes.data

    valid = (
        (lons >= lon_min) * (lons <= lon_max) *
        (lats >= lat_min) * (lats <= lat_max)
    )
    if max_time_diff is not None and "time" in data:
        dtype = data.time.dtype
        if "latitude" in data.time.dims:
            time = data.time.astype(np.int64).interp(latitude=latitudes, longitude=longitudes, method="nearest")
            time = time.astype(dtype)
        else:
            time = data.time
        time_diff = time - sensor_time
        valid = valid * (time_diff.data < max_time_diff)

    scan_inds, pixel_inds = np.where(valid)


    for scan_ind, pix_ind in zip(scan_inds, pixel_inds):

        lon_p = longitudes.data[scan_ind, pix_ind]
        lat_p = latitudes.data[scan_ind, pix_ind]
        lon_s = sensor_longitudes.data[scan_ind, pix_ind]
        lat_s = sensor_latitudes.data[scan_ind, pix_ind]
        alt_s = sensor_altitudes.data[scan_ind, pix_ind]

        lon_range = np.abs(lons_data - lon_p) < 0.5 * area_of_influence
        lat_range = np.abs(lats_data - lat_p) < 0.5 * area_of_influence

        if "longitude" in data.dims:
            data_fp = data[{"longitude": lon_range, "latitude": lat_range}]
        else:
            data_fp = data[{data.surface_precip.dims[0]: lon_range * lat_range}]

        lons_d = data_fp.longitude.data
        lats_d = data_fp.latitude.data
        if lons_d.ndim == 1:
            lons_d, lats_d = np.meshgrid(lons_d, lats_d)

        wgts = calculate_footprint_weights(
            lons_d,
            lats_d,
            (lon_p, lat_p),
            (lon_s, lat_s, alt_s),
            beam_width=beam_width
        )

        for var in data_fp:
            if var == "time":
                continue
            wgts_var = wgts.__getitem__((...,) + (None,) * (data_fp[var].data.ndim - wgts.ndim))
            valid_mask = np.isfinite(data_fp[var].data)
            fp_sum = (wgts_var * np.where(valid_mask, data_fp[var].data, 0.0)).sum((0, 1))
            wgt_sum = np.where(valid_mask, wgts_var, 0.0).sum((0, 1))
            results[var][-1][scan_ind, pix_ind] = fp_sum / wgt_sum

    return xr.Dataset(results)


def interp_along_swath(
        ref_data: xr.Dataset,
        scan_time: xr.DataArray,
        dimension: str = "time"
) -> xr.Dataset:
    """
    Interpolate time-gridded data to swath.

    Helper function that interpolates data with an independent time dimension
    to swath data, where the time varies by scan.

    Args:
        ref_data: An xarray.Dataset containing the data to interpolate.
        scan_time: An xarray.DataArray containing the scan times to which
            to interpolate 'ref_data'.
        dimension: The name of the time dimension along which to
            interpolate.

    Return:
        The interpolated dataset.
    """
    if ref_data.time.size == 1:
        return ref_data[{"time": 0}]

    time_bins = ref_data.time.data
    time_bins = np.concatenate(
        [
            [time_bins[0] - 0.5 * (time_bins[1] - time_bins[0])],
            time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1]),
            [time_bins[-1] + 0.5 * (time_bins[-1] - time_bins[-2])],
        ],
        axis=0,
    )
    scan_time = scan_time.astype(ref_data.time.dtype)
    inds = np.digitize(scan_time.astype(np.int64), time_bins.astype(np.int64))
    out_of_range = (inds == 0) + (inds==time_bins.size)
    inds[out_of_range] = time_bins.size // 2
    inds = xr.DataArray(inds - 1, dims=(("latitude", "longitude")))

    time = ref_data.time[{"time": inds}]
    ref_data_r = ref_data[{"time": inds}]

    ref_data_r["time"] = time
    invalid_time = np.isnan(scan_time.data)
    ref_data_r["time"].data[invalid_time] = np.datetime64("NaT")

    return ref_data_r
