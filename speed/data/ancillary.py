"""
speed.data.ancillary
====================

Functionality to extract ancillary data for SPEED collocations.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
import struct
from typing import List, Optional, Tuple

import click
import numpy as np
from pansat.time import TimeRange, to_datetime, to_datetime64
from pansat.geometry import LonLatRect
from pansat.products.dem.globe import globe
from rich.progress import track, Progress
from scipy.ndimage import binary_dilation
import unlzw3
import xarray as xr


import speed


LOGGER = logging.getLogger(__name__)


FOOTPRINTS = {
    "gmi": 32,
    "atms": 74
}


def find_era5_sfc_files(
        era5_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
) -> List[Path]:
    """
    Find ERA5 surface-data files to load within a given time range.

    Args:
        era5_path: The path containing the ERA5 files.
        start_time: The start time of the time range for which to load
            ERA5 data.
        end_time: The end time of the time range for which to load
            ERA5 data.

    Return:
        A list of path object pointing to the ERA5 files containing the
        data covering the given time range.
    """
    era5_path = Path(era5_path)
    start_time = to_datetime(start_time)
    end_time = to_datetime(end_time)

    era5_files = []
    time = datetime(start_time.year, start_time.month, day=1)
    while time <= end_time:
        mpath = era5_path / f"{time.year}{time.month:02}"
        era5_files += sorted(list(mpath.glob("ERA5*surf.nc")))
        _, days = monthrange(time.year, time.month)
        time += timedelta(days=days)

    filtered = []
    for era5_file in era5_files:
        file_start = datetime.strptime(era5_file.name.split("_")[1], "%Y%m%d")
        file_end = file_start + timedelta(days=1)
        if not (end_time < file_start or start_time > file_end):
            filtered.append(era5_file)

    return filtered


def find_era5_lai_files(
        era5_lai_path: Path,
        start_time: np.datetime64,
        end_time: np.datetime64,
) -> List[Path]:
    """
    Find ERA5 leaf-area index (LAI) files to load within a given time range.

    Args:
        era5_lai_path: The path containing the ERA5 files.
        start_time: The start time of the time range for which to load
            ERA5 data.
        end_time: The end time of the time range for which to load
            ERA5 data.

    Return:
        A list of path object pointing to the ERA5 files containing the
        data covering the given time range.
    """
    era5_lai_path = Path(era5_lai_path)
    start_time = to_datetime(start_time)
    end_time = to_datetime(end_time)

    lai_paths = []
    time = datetime(start_time.year, start_time.month, day=1)
    while time <= end_time:
        mpath = era5_lai_path / f"{time.year}{time.month:02}"
        lai_paths += sorted(list(mpath.glob("ERA5*lai.nc")))
        _, days = monthrange(time.year, time.month)
        time += timedelta(days=days)

    filtered = []
    for lai_path in lai_paths:
        file_start = datetime.strptime(lai_path.name.split("_")[1], "%Y%m%d")
        file_end = file_start + timedelta(days=1)
        if not (end_time < file_start or start_time > file_end):
            filtered.append(lai_path)

    return filtered


def load_era5_ancillary_data(
        era5_sfc_files: List[Path],
        era5_lai_files: List[Path],
        roi: Optional[Tuple[float, float, float, float]] = None
) -> xr.Dataset:
    """
    Load ancillary data from ERA5 files.

    Args:
         era5_sfc_files: A list containing the ERA5 surface-data files from which to load the data.
         era5_lai_files: A list containing the ERA5 leaf-area-index files from which to load the data.

    Return:
         An xarray.Dataset containing the ERA5 surface ancillary combined with
         the leaf-area index.
    """
    new_names = {
        "u10": "ten_meter_wind_u",
        "v10": "ten_meter_wind_v",
        "d2m": "two_meter_dew_point",
        "t2m": "two_meter_temperature",
        "cape": "cape",
        "siconc": "sea_ice_concentration",
        "sst": "sea_surface_temperature",
        "skt": "skin_temperature",
        "sd": "snow_depth",
        "sf": "snowfall",
        "sp": "surface_pressure",
        "tciw": "total_column_cloud_ice_water",
        "tclw": "total_column_cloud_liquid_water",
        "tcwv": "total_column_water_vapor",
        "tp": "total_precipitation",
        "cp": "convective_precipitation",
    }

    data = []
    for era5_file in era5_sfc_files:
        with xr.open_dataset(era5_file) as inpt:
            if roi is not None:
                lon_min, lat_min, lon_max, lat_max = roi
                lon_min = (lon_min + 360) % 360
                lon_max = (lon_max + 360) % 360
                if lon_min > lon_max:
                    lon_min, lon_max = lon_max, lon_min
                lons = inpt.longitude.data
                lats = inpt.latitude.data
                lon_min -= 0.5
                lat_min -= 0.5
                lon_max += 0.5
                lat_max += 0.5
                lon_mask = (lon_min <= lons) * (lons <= lon_max)
                lat_mask = (lat_min <= lats) * (lats <= lat_max)
                inpt = inpt[{"longitude": lon_mask, "latitude": lat_mask}]
            data.append(inpt[list(new_names.keys())].rename(**new_names))

    data = xr.concat(data, "time")

    lai_data = []
    for era5_file in era5_lai_files:
        with xr.open_dataset(era5_file) as inpt:
            if roi is not None:
                lon_min, lat_min, lon_max, lat_max = roi
                lon_min = (lon_min + 360) % 360
                lon_max = (lon_max + 360) % 360
                if lon_min > lon_max:
                    lon_min, lon_max = lon_max, lon_min
                lons = inpt.longitude.data
                lats = inpt.latitude.data
                lon_min -= 0.5
                lat_min -= 0.5
                lon_max += 0.5
                lat_max += 0.5
                lon_mask = (lon_min <= lons) * (lons <= lon_max)
                lat_mask = (lat_min <= lats) * (lats <= lat_max)
                inpt = inpt[{"longitude": lon_mask, "latitude": lat_mask}]
            lai_data.append(inpt[["lai_hv", "lai_lv"]])

    lai_data = xr.concat(lai_data, dim="time")
    data["leaf_area_index"] = lai_data["lai_hv"] + lai_data["lai_lv"]

    lons = data.longitude.data
    lons[lons > 180] -= 360

    return data.sortby("longitude")


def find_autosnow_file(
        autosnow_path: Path,
        date: np.datetime64
) -> Path:
    """
    Find autosnow files for a given day.

    Args:
        autosnow_path: The root directory of the folder tree containing the autosnow files.
        date: A date object specifying the day for which to load the autosnow data.

    Return:
        A path pointing to the autosnow file for the given day.
    """
    autosnow_path = Path(autosnow_path)
    date = to_datetime(date)
    path = autosnow_path / date.strftime("autosnowV3/%Y/gmasi_snowice_reproc_v003_%Y%j.Z")
    return path


def load_autosnow_data(path: Path) -> xr.DataArray:
    """
    Load autosnow data into a data array.

    Args:
         path: A path object pointing to an AutoSnow file.

    Return:
         An xr.DataArray containing the data loaded from the file.
    """
    n_lons = 9000
    n_lats = 4500
    with open(path, 'rb') as inpt:
            cmpd = inpt.read()
    dcmpd = unlzw3.unlzw(cmpd)
    data = np.frombuffer(dcmpd, dtype="i1")
    data = data.reshape((n_lats, n_lons))
    lons = np.linspace(-180, 180, n_lons + 1)
    lons = 0.5 * (lons[1:] + lons[:-1])
    lats = np.linspace(90, -90, n_lats + 1)
    lats = 0.5 * (lats[1:] + lats[:-1])
    return xr.DataArray(
        data=data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": lats,
            "longitude": lons
        }
    )


def load_landmask_data(
        ancillary_dir: Path,
        footprint: str = "32",
        resolution: int = 32
) -> xr.DataArray:
    """
    Load GPM landmask file into a data array.

    Args:
         path: A path object pointing to the preprocessor ancillary directory.

    Return:
         An xr.DataArray containing the data loaded from the file.
    """
    n_lons = 360 * resolution
    n_lats = 180 * resolution
    landmask_file = ancillary_dir / f"landmask{footprint}_{resolution}.bin"
    data = np.fromfile(landmask_file, dtype="i1").reshape((n_lats, n_lons))
    lons = np.linspace(-180, 180, n_lons + 1)
    lons = 0.5 * (lons[1:] + lons[:-1])
    lats = np.linspace(90, -90, n_lats + 1)
    lats = 0.5 * (lats[1:] + lats[:-1])
    return xr.DataArray(
        data=data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": lats,
            "longitude": lons
        }
    )


def load_emissivity_data(ancillary_dir: Path, date: np.datetime64) -> xr.DataArray:
    """
    Load GPM emissivity data into file.

    Args:
         path: A path object pointing to the preprocessor ancillary directory.
         date: A date object defining the month for which to load to emissivity data.

    Return:
         An xr.DataArray containing the data loaded from the file.
    """
    month = to_datetime(date).month
    emiss_file = ancillary_dir / f"emiss_class_{month:02}.dat"

    n_lons = 720
    n_lats = 359

    data = np.fromfile(emiss_file, dtype="i2").reshape((n_lats, n_lons))

    data_c = data[1:-1, 1:-1]
    for ind in range(3):
        for x_shift in [-1, 1]:
            for y_shift in [-1, 1]:
                data_shifted = data[1 + y_shift: n_lats - 1 + y_shift, 1 + x_shift : n_lons - 1 + x_shift]
                update_mask = (data_c == 0) * (data_shifted != 0)
                data_c[update_mask] = data_shifted[update_mask]

    lons = np.linspace(-180, 180, n_lons + 1)
    lons = 0.5 * (lons[1:] + lons[:-1])
    lats = np.linspace(90, -90, n_lats + 1)
    lats = 0.5 * (lats[1:] + lats[:-1])
    return xr.DataArray(
        data=data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": lats,
            "longitude": lons
        }
    )


def load_emissivity_data_all_months(ancillary_dir: Path) -> xr.DataArray:
    """
    Load GPM emissivity data for all months.

    Args:
         path: A path object pointing to the preprocessor ancillary directory.

    Return:
         An xr.DataArray containing the data loaded from the file.
    """
    n_lons = 720
    n_lats = 359

    emiss_data = []

    for month in range(1, 13):
        emiss_file = ancillary_dir / f"emiss_class_{month:02}.dat"
        data = np.fromfile(emiss_file, dtype="i2").reshape((n_lats, n_lons))

        data_c = data[1:-1, 1:-1]
        for ind in range(3):
            for x_shift in [-1, 1]:
                for y_shift in [-1, 1]:
                    data_shifted = data[1 + y_shift: n_lats - 1 + y_shift, 1 + x_shift : n_lons - 1 + x_shift]
                    update_mask = (data_c == 0) * (data_shifted != 0)
                    data_c[update_mask] = data_shifted[update_mask]

        lons = np.linspace(-180, 180, n_lons + 1)
        lons = 0.5 * (lons[1:] + lons[:-1])
        lats = np.linspace(90, -90, n_lats + 1)
        lats = 0.5 * (lats[1:] + lats[:-1])
        emiss_data.append(xr.DataArray(
            data=data,
            dims=("latitude", "longitude"),
            coords={
                "latitude": lats,
                "longitude": lons
            }
        ))
    return xr.concat(emiss_data, dim="month")


def load_mountain_mask(ancillary_dir: Path) -> xr.DataArray:
    """
    Read GPROF mountain mask.

    Args:
        ancillary_dir: A pathlib.Path object pointing to the GPROF preprocessor ancillary directory

    Return:
        An xarray.DataArray containing the mountain mask.
    """
    fname = Path(ancillary_dir) / "k3classes_0.1deg.asc.bin"
    with open(fname, "rb") as data:
        n_lons = struct.unpack("i", data.read(4))[0]
        n_lons = struct.unpack("i", data.read(4))[0]
        n_lats = struct.unpack("i", data.read(4))[0]
        data = np.fromfile(data, dtype=np.int32, count=n_lons * n_lats)

    data = data.reshape(n_lats, n_lons)
    data = np.roll(data, shift=n_lons // 2, axis=-1)

    lons = np.linspace(-180, 180, n_lons + 1)
    lons = 0.5 * (lons[1:] + lons[:-1])
    lats = np.linspace(-90, 90, n_lats + 1)
    lats = 0.5 * (lats[1:] + lats[:-1])

    data = xr.DataArray(
        data=data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": lats,
            "longitude": lons
        }
    )
    return data


def load_gprof_surface_type_data(
        ancillary_dir: Path,
        ingest_dir: Path,
        date: np.datetime64,
        footprint: str = "32",
        resolution: int = 32,
        roi: Optional[Tuple[float, float, float, float]] = None
) -> xr.DataArray:
    """
    Load the CSU surface type classification employed by GPROF.

    Args:
        ancillary_dir: Path pointing to the directory containing the GPROF ancillary data.
        ingest_dir: Path pointing to the directory containing the GPROF ingest data.
        date: A np.datetime64 object specifying the date for which to load the surface
            type classes.
        roi: An optional tuple ``(lon_min, lat_min, lon_max, lat_max)`` defining a bounding


    Return:
        An xarray.Dataset containing the GPROF surface classification.
    """
    ancillary_dir = Path(ancillary_dir)
    ingest_dir = Path(ingest_dir)

    date = to_datetime(date)
    month = date.month

    landmask_data = load_landmask_data(
        ancillary_dir,
        footprint=footprint,
        resolution=resolution
    )
    lons = landmask_data.longitude.data
    lats = landmask_data.latitude.data
    if roi is not None:
        lon_min, lat_min, lon_max, lat_max = roi
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        landmask_data = landmask_data[{"longitude": lon_mask, "latitude": lat_mask}]

    emissivity_data = load_emissivity_data_all_months(ancillary_dir)
    if roi is not None:
        lons = emissivity_data.longitude.data
        lats = emissivity_data.latitude.data
        lon_min, lat_min, lon_max, lat_max = roi
        lon_min -= 1.0
        lon_max += 1.0
        lat_min -= 1.0
        lat_max += 1.0
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        emissivity_data = emissivity_data[{"longitude": lon_mask, "latitude": lat_mask}]
    emissivity_data_all = emissivity_data.interp(
        longitude=landmask_data.longitude.data,
        latitude=landmask_data.latitude.data,
        method="nearest"
    )
    emissivity_data = emissivity_data_all[{"month": month - 1}]

    autosnow_file = find_autosnow_file(ingest_dir, date)
    autosnow_data = load_autosnow_data(autosnow_file).copy()
    if roi is not None:
        lons = autosnow_data.longitude.data
        lats = autosnow_data.latitude.data
        lon_min, lat_min, lon_max, lat_max = roi
        lon_min -= 1.0
        lon_max += 1.0
        lat_min -= 1.0
        lat_max += 1.0
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        autosnow_data = autosnow_data[{"longitude": lon_mask, "latitude": lat_mask}]

    # Expand sea ice
    sea_ice_mask = autosnow_data.data == 3
    sea_ice_mask = binary_dilation(sea_ice_mask, structure=np.ones((10, 10)))
    zero_mask = autosnow_data.data == 0
    autosnow_data.data[zero_mask * sea_ice_mask] = 3

    autosnow_data = autosnow_data.interp(
        latitude=landmask_data.latitude.data,
        longitude=landmask_data.longitude.data,
        method="nearest"
    )

    mountain_mask = load_mountain_mask(ancillary_dir)
    if roi is not None:
        lons = mountain_mask.longitude.data
        lats = mountain_mask.latitude.data
        lon_min, lat_min, lon_max, lat_max = roi
        lon_min -= 1.0
        lon_max += 1.0
        lat_min -= 1.0
        lat_max += 1.0
        lon_mask = (lon_min <= lons) * (lons <= lon_max)
        lat_mask = (lat_min <= lats) * (lats <= lat_max)
        mountain_mask = mountain_mask[{"longitude": lon_mask, "latitude": lat_mask}]
    mountain_mask = mountain_mask.interp(
        latitude=landmask_data.latitude.data,
        longitude=landmask_data.longitude.data
    )

    sfc_type = np.zeros((landmask_data.shape), dtype=np.int8)

    sfc_type[(landmask_data >= 0) * (landmask_data <= 2)] = 10
    sfc_type[(landmask_data > 2) * (landmask_data <= 25)] = 30
    sfc_type[(landmask_data > 25) * (landmask_data <= 75)] = 31
    sfc_type[(landmask_data > 75) * (landmask_data <= 95)] = 32
    sfc_type[(landmask_data > 95)] = 20

    snow = autosnow_data.data

    sfc_type[sfc_type == 10] = 1
    sfc_type[(sfc_type == 1) * (snow == 2)] = 2
    sfc_type[snow == 3] = 2
    sfc_type[snow == 5] = 16

    land = sfc_type == 20
    snow_pixels = ((snow == 2) + (snow == 3))
    no_snow_pixels = ~snow_pixels

    mask = (emissivity_data.data >= 6) * (emissivity_data.data <= 9) * snow_pixels * land
    sfc_type[mask] = emissivity_data.data[mask] + 2

    mask = ((emissivity_data.data >= 1) * (emissivity_data.data <= 5) + (emissivity_data.data == 10)) * snow_pixels * land
    sfc_type[mask] = 10

    mask = (emissivity_data.data == 0) * snow_pixels * land
    sfc_type[mask] = 8

    mask = (emissivity_data.data >= 1) * (emissivity_data.data <= 5) * no_snow_pixels * land
    sfc_type[mask] = emissivity_data.data[mask] + 2

    frozen = (emissivity_data.data >= 6) * (emissivity_data.data <= 9) * land * no_snow_pixels
    for ind in range(11):
        prev_month = (month - ind - 1) % 12
        prev_emiss = emissivity_data_all[{"month": prev_month}]
        mask = frozen * (prev_emiss < 6) * (sfc_type > 7) * (sfc_type != 12)
        sfc_type[mask] = prev_emiss.data[mask] + 2
        mask = frozen * (prev_emiss == 10) * (sfc_type != 12) * (sfc_type > 8)
        sfc_type[mask] = 12

    mask = (emissivity_data.data == 10) * no_snow_pixels * land
    sfc_type[mask] = 12

    mask = (emissivity_data.data == 0) * land * no_snow_pixels
    sfc_type[mask] = 1

    sfc_type[sfc_type == 20] = 12

    sfc_type[sfc_type == 30] = 13
    sfc_type[sfc_type == 31] = 14
    sfc_type[sfc_type == 32] = 15

    mask = (sfc_type >= 13) * (sfc_type <= 15)
    sfc_type[mask * (snow == 2)] = 10
    sfc_type[mask * (snow == 3)] = 2

    mtn_no_snow = (sfc_type >= 3) * (sfc_type <= 7) * (mountain_mask >= 1)
    sfc_type[mtn_no_snow] = 17
    mtn_snow = (sfc_type >= 8) * (sfc_type <= 11) * (mountain_mask >= 1)
    sfc_type[mtn_snow] = 18

    return xr.DataArray(
        data=sfc_type,
        dims=("latitude", "longitude"),
        coords={
            "latitude": landmask_data.latitude.data,
            "longitude": landmask_data.longitude.data
        }
    )


def load_elevation_data(roi: Tuple[float, float]) -> xr.Dataset:
    """
    Load elevation data from the NOAA GLOBE dataset.
    """
    date = np.datetime64("2020-01-01")
    roi = LonLatRect(*roi)
    recs = globe.get(time_range=TimeRange(date), roi=roi)
    tiles = {}
    for rec in recs:
        ind = ord(rec.filename[0]) - ord('a')
        row_ind = ind // 4
        col_ind = ind % 4
        tiles[(row_ind, col_ind)] = globe.open(rec)

    row_inds = np.unique([coord[0] for coord in tiles.keys()])
    col_inds = np.unique([coord[1] for coord in tiles.keys()])

    combined = []
    for row_ind in row_inds:
        nested = []
        for col_ind in col_inds:
            nested.append(tiles[row_ind, col_ind])
        nested = xr.concat(nested, dim="longitude")
        combined.append(nested)
    return xr.concat(combined, dim="latitude")


def add_ancillary_data(
        path_on_swath: Path,
        path_gridded: Path,
        era5_path: Path,
        era5_lai_path: Path,
        gprof_ancillary_dir: Path,
        gprof_ingest_dir: Path
) -> None:
    """
    Add ancillary data to SPEED collocations.

    The data is added to both the 'on-swath' and 'gridded' files in a separate group
    called 'ancillary'.

    Args:
        path_on_swath: Path to the file containing the collocations extracted in on_swath format.
        path_gridded: Path to the file containing the collocations extract in gridded format.
        era5_path: Path pointing to the folder containing the ERA5 files.
        era5_lai_path: Path pointing to the folder containing the ERA5 LAI data.
        gprof_ancillary_dir: Path pointing to the GPROF ancillary dir.
        gprof_ingest_dir: Path pointing to the GPROF ingest dir.
    """
    sensor = path_on_swath.name.split("_")[-2]
    time_str = path_on_swath.name.split("_")[-1][:-3]
    median_time = to_datetime64(datetime.strptime(time_str, "%Y%m%d%H%M%S"))

    try:
        data = xr.open_dataset(path_on_swath, group="ancillary_data")
        data.close()
        LOGGER.info(
            "Skipping input files %s because they already contain ancillary_data observations.",
            path_on_swath
        )
        return None
    except (KeyError, ValueError):
        # No ancillary_data group exists yet, proceed with processing
        LOGGER.debug("No existing ancillary data found in %s, proceeding with processing", path_on_swath)
    except Exception as e:
        LOGGER.warning("Error checking for existing ancillary data in %s: %s", path_on_swath, e)
        # Continue processing despite check failure

    with xr.load_dataset(path_on_swath, group="input_data") as data_on_swath:
        lons_os = data_on_swath.longitude
        lon_min, lon_max = lons_os.data.min(), lons_os.data.max()
        lats_os = data_on_swath.latitude
        lat_min, lat_max = lats_os.data.min(), lats_os.data.max()

        roi = (lon_min, lat_min, lon_max, lat_max)
        scan_time = data_on_swath.scan_time
    data_on_swath.close()

    # Store ancillary data in gridded geometry
    with xr.load_dataset(path_gridded, group="reference_data") as data_gridded:
        # Batch compute operations for better memory efficiency
        coords = data_gridded[['longitude', 'latitude', 'time']].compute()
        lons_g, lats_g, time_g = coords.longitude, coords.latitude, coords.time
    data_gridded.close()

    start_time = scan_time.data.min()
    end_time = scan_time.data.max()
    era5_sfc_files = find_era5_sfc_files(era5_path, start_time, end_time)
    era5_lai_files = find_era5_lai_files(era5_lai_path, start_time, end_time)
    era5_data = load_era5_ancillary_data(
        era5_sfc_files, era5_lai_files, roi=roi
    )
    surface_type = load_gprof_surface_type_data(
        ancillary_dir=gprof_ancillary_dir,
        ingest_dir=gprof_ingest_dir,
        date=median_time,
        footprint=FOOTPRINTS[sensor.lower()],
        resolution=32,
        roi=roi
    )
    elevation = load_elevation_data(roi=roi)

    # Store ancillary data in on-swath geometry
    ancillary_data = era5_data.interp(
        latitude=lats_os,
        longitude=lons_os,
        time=scan_time
    ).rename(time="scan_time")
    surface_type_os = surface_type.interp(
        latitude=lats_os,
        longitude=lons_os,
        method="nearest",
        kwargs={"fill_value": -1}
    )
    ancillary_data["surface_type"] = surface_type_os
    elevation_os = elevation.interp(
        latitude=lats_os,
        longitude=lons_os,
    )
    ancillary_data["elevation"] = elevation_os["elevation"]
    encoding = {
        var: {"dtype": "float32", "compression": "zstd"}
        for var in ancillary_data.variables if var != "surface_type"
    }
    encoding["surface_type"] = {"dtype": "int8", "compression": "zstd"}
    ancillary_data.to_netcdf(
        path_on_swath,
        group="ancillary_data",
        mode="a",
    )

    ancillary_data = era5_data.interp(
        latitude=lats_g,
        longitude=lons_g,
        time=time_g,
    )
    surface_type_g = surface_type.interp(
        latitude=lats_g,
        longitude=lons_g,
        method="nearest",
        kwargs={"fill_value": -1}
    )
    ancillary_data["surface_type"] = surface_type_g
    elevation_g = elevation.interp(
        latitude=lats_g,
        longitude=lons_g,
    )
    ancillary_data["elevation"] = elevation_g["elevation"]
    encoding = {
        var: {"dtype": "float32", "compression": "zstd"}
        for var in ancillary_data.variables if var != "surface_type"
    }
    encoding["surface_type"] = {"dtype": "int8", "compression": "zstd"}
    ancillary_data.to_netcdf(
        path_gridded,
        group="ancillary_data",
        mode="a",
    )


@click.command()
@click.argument("collocation_path", type=str)
@click.option("--n_processes", type=int, default=1)
@click.option("--pattern", type=str, default="*.nc")
@click.option("--era5_path", type=str, default="/qdata2/archive/ERA5")
@click.option("--era5_lai_path", type=str, default="/pdata4/pbrown/ERA5")
@click.option("--gprof_ancillary_dir", type=str, default="/qdata1/pbrown/gpm/ppancillary")
@click.option("--gprof_ingest_dir", type=str, default="/qdata1/pbrown/gpm/ppingest")
def cli(
        collocation_path: str,
        n_processes: int = 1,
        pattern: str = "*.nc",
        era5_path: str = "/qdata2/archive/ERA5",
        era5_lai_path: str = "/pdata4/pbrown/ERA5",
        gprof_ancillary_dir: str = "/qdata1/pbrown/gpm/ppancillary",
        gprof_ingest_dir: str = "/qdata1/pbrown/gpm/ppingest"
):
    """
    Extract ancillary data for GPM collocations.

    speed extract_ancillary_data collocation_path

    Extracts GPROF ancillary data for GPM collocations scenes.
    """
    collocation_path = Path(collocation_path)
    if not collocation_path.exists():
        LOGGER.error("Provided collocation path must point to an existing directory.")
        return 1

    files_on_swath = sorted(list((collocation_path / "on_swath").glob(pattern)))
    files_gridded = sorted(list((collocation_path / "gridded").glob(pattern)))

    times_on_swath = {}
    for f_on_swath in files_on_swath:
        time_str = f_on_swath.name.split("_")[-1][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_on_swath[median_time] = f_on_swath

    times_gridded = {}
    for f_gridded in files_gridded:
        time_str = f_gridded.name.split("_")[-1][:-3]
        median_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
        times_gridded[median_time] = f_gridded

    combined = set(times_gridded.keys()).intersection(set(times_on_swath.keys()))

    LOGGER.info(f"Found {len(combined)} collocations in {collocation_path}.")

    if n_processes < 2:
        for median_time in track(
                combined,
                description="Extracting ancillary data:",
                console=speed.logging.get_console()
        ):
            try:
                add_ancillary_data(
                    times_on_swath[median_time],
                    times_gridded[median_time],
                    era5_path=era5_path,
                    era5_lai_path=era5_lai_path,
                    gprof_ancillary_dir=gprof_ancillary_dir,
                    gprof_ingest_dir=gprof_ingest_dir
                )
            except Exception:
                LOGGER.exception(
                    "Processing of the collocation with median time %s failed "
                    "with the following error.",
                    median_time
                )
    else:
        pool = ProcessPoolExecutor(
            max_workers=n_processes
        )
        tasks = []
        for median_time in combined:
            tasks.append(
                pool.submit(
                    add_ancillary_data,
                    times_on_swath[median_time],
                    times_gridded[median_time],
                    era5_path=era5_path,
                    era5_lai_path=era5_lai_path,
                    gprof_ancillary_dir=gprof_ancillary_dir,
                    gprof_ingest_dir=gprof_ingest_dir
                )
            )
        with Progress(console=speed.logging.get_console()) as progress:
            extraction = progress.add_task("Extracting ancillary data:", total=len(tasks))
            for task in as_completed(tasks):
                try:
                    task.result()
                except Exception:
                    LOGGER.exception(
                        "The following error was encountered when processing collocation "
                        "with median time %s.",
                        median_time
                    )
                progress.advance(extraction, advance=1.0)
