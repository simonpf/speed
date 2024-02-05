"""
speed.data.utils
================

Utility functions for data processing.
"""
from pathlib import Path
from typing import Optional

from gprof_nn.data.l1c import L1CFile
from pansat.time import to_datetime

import numpy as np
from pansat import Granule
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr


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


def extract_scans(granule: Granule, dest: Path) -> Path:
    """
    Extract and write scans from L1C file into a separate file.

    Args:
        granule: A pansat granule specifying a subset of an orbit.
        dest: A directory to which the extracted scans will be written.

    Return:
        The path of the file containing the extracted scans.
    """
    scan_start, scan_end = granule.primary_index_range
    l1c_path = granule.file_record.local_path
    l1c_file = L1CFile(granule.file_record.local_path)
    output_filename = dest / l1c_path.name
    l1c_file.extract_scan_range(scan_start, scan_end, output_filename)
    return output_filename


def save_data_native(
    sensor_name: str,
    reference_data_name: str,
    preprocessor_data: xr.Dataset,
    reference_data: xr.Dataset,
    output_path: Path,
    min_scans: int = 96,
) -> None:
    """
    Save collocations in native format.

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
    native_path = output_path / "native"
    native_path.mkdir(exist_ok=True)

    # Determine number of valid reference pixels in scene.
    surface_precip = reference_data.surface_precip.data

    valid_scans = np.where(np.any(np.isfinite(surface_precip), -1))[0]
    scan_start = valid_scans[0]
    scan_end = valid_scans[-1]
    n_scans = scan_end - scan_start
    if n_scans < min_scans:
        scan_c = int(0.5 * (scan_end + scan_start))
        scan_start = max(scan_c - min_scans // 2, 0)
        scan_end = min(scan_start + min_scans, surface_precip.shape[0])

    preprocessor_data = preprocessor_data[{"scans": slice(scan_start, scan_end)}]
    reference_data = reference_data[{"scans": slice(scan_start, scan_end)}]

    time = to_datetime(preprocessor_data.scan_time.mean().data)
    fname = time.strftime(f"{reference_data_name}_{sensor_name}_%Y%m%d%H%M%S.nc")

    output_file = native_path / fname
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
        time: The mean scan-time used to save the collocation in native
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
    gridded_path.mkdir(exist_ok=True)

    if reference_data.longitude[0] < -179 and reference_data.longitude[-1] > 179:

        surface_precip = reference_data.surface_precip.data
        tbs = preprocessor_data.tbs_mw.data
        valid = np.isfinite(surface_precip) * np.isfinite(tbs).any(-1)
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
    tbs = preprocessor_data.tbs_mw.data
    valid = np.isfinite(surface_precip) * np.isfinite(tbs).any(-1)
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
    preprocessor_data.to_netcdf(output_file, group="input_data")
    reference_data.to_netcdf(output_file, group="reference_data", mode="a")


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
    lons_t, lats_t = target_grid.grid.get_lonlats()
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

    dims = ("scans", "pixels")
    resampled = {}

    row_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", swath.shape, row_inds, ind_in, ind_out, inds
    )
    col_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", swath.shape, col_inds, ind_in, ind_out, inds
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
    lons_t, lats_t = target_grid.grid.get_lonlats()
    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )

    swath = SwathDefinition(lons=lons, lats=lats)
    target = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    pixel_inds, scan_inds = np.meshgrid(dataset.pixels.data, dataset.scans.data)

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    dims = ("latitude", "longitude")
    resampled = {}

    scan_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", target.shape, scan_inds, ind_in, ind_out, inds, fill_value=-1
    )
    scan_inds_gridded = -np.ones(target_grid.grid.shape, dtype=np.int16)
    scan_inds_gridded[valid_pixels] = scan_inds_r

    pixel_inds_r = kd_tree.get_sample_from_neighbour_info(
        "nn", target.shape, pixel_inds, ind_in, ind_out, inds, fill_value=-1
    )
    pixel_inds_gridded = -np.ones(target_grid.grid.shape, dtype=np.int16)
    pixel_inds_gridded[valid_pixels] = pixel_inds_r

    return xr.Dataset(
        {
            "scan_index": (dims, scan_inds_gridded),
            "pixel_index": (dims, pixel_inds_gridded),
        }
    )



def extract_scenes(
        input_data: xr.Dataset,
        reference_data: xr.Dataset,
        output_folder: Path,
        size: int = 256,
        overlap: float = 0.0,
        filename_pattern = "collocation_{time}",
        min_input_frac: Optional[float] = None
):
    """
    Extract scenes of a given size from collocations.

    Args:
        input_data: An xarray.Dataset containing the retrievl input data.
        reference_data: An xarray.Dataset containing the correpsonding reference data.
        output_folder: The folder to which to write the extracted scenes.
        size: The size of the scenes.
        overlap: The maximum overlap in any direction between two scenes.
    """

    spatial_dims = input_data.tbs_mw.dims[:2]

    valid_input = np.any(np.isfinite(input_data.tbs_mw.data), -1)
    valid_output = np.isfinite(reference_data.surface_precip.data)
    valid = valid_input * valid_output

    n_rows = input_data[spatial_dims[0]].size
    n_cols = input_data[spatial_dims[1]].size

    row_inds, col_inds = np.where(valid)
    within = (
        (row_inds >= size // 2) * (row_inds < n_rows - size // 2) *
        (col_inds >= size // 2) * (col_inds < n_cols - size // 2)
    )
    row_inds = row_inds[within]
    col_inds = col_inds[within]

    while len(row_inds) > 0:

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

        scan_times = inpt.scan_time.data
        scan_times = scan_times[np.isfinite(scan_times)]
        time = scan_times[0] + np.median(scan_times - scan_times[0])
        time = to_datetime(time)
        if filename_pattern.endswith(".nc"):
            filename_pattern = filename_pattern[:-3]
        filename = filename_pattern.format(time=time.strftime("%Y%m%d%H%M%S")) + ".nc"

        ref = ref.drop_vars(["latitude", "longitude"])
        dataset = xr.merge([inpt, ref])
        dataset.to_netcdf(output_folder / filename)

        margin = max((1.0 - 0.0) * size, 1)
        covered = (
            (row_inds >= c_row - margin) * (row_inds <= c_row + margin) *
            (col_inds >= c_col - margin) * (col_inds <= c_col + margin)
        )

        n_inds = len(row_inds)

        row_inds = row_inds[~covered]
        col_inds = col_inds[~covered]

        assert n_inds > len(row_inds)
