"""
Tests for the speed.data.ancillary module.
"""
from pathlib import Path
import pytest

import numpy as np

from speed.data.ancillary import (
    find_era5_sfc_files,
    find_era5_lai_files,
    load_era5_ancillary_data,
    find_autosnow_file,
    load_autosnow_data,
    load_landmask_data,
    load_emissivity_data,
    load_gprof_surface_type_data
)


ERA5_SFC_PATH = Path("/qdata2/archive/ERA5")
ERA5_LAI_PATH = Path("/pdata4/pbrown/ERA5")
HAS_ERA5_DATA = ERA5_SFC_PATH.exists() and ERA5_LAI_PATH.exists()
NEEDS_ERA5_DATA = pytest.mark.skipif(
    not HAS_ERA5_DATA,
    reason="Needs ERA5 data."
)

INGEST_DIR = Path("/qdata1/pbrown/gpm/ppingest")
HAS_INGEST_DIR = INGEST_DIR.exists()
NEEDS_INGEST_DATA = pytest.mark.skipif(
    not HAS_INGEST_DIR,
    reason="Needs preprocessor ingest dir."
)

ANCILLARY_DIR = Path("/qdata1/pbrown/gpm/ppancillary")
HAS_ANCILLARY_DIR = ANCILLARY_DIR.exists()
NEEDS_ANCILLARY_DATA = pytest.mark.skipif(
    not HAS_ANCILLARY_DIR,
    reason="Needs preprocessor ancillary dir."
)

@NEEDS_ERA5_DATA
def test_find_era5_sfc_files():
    """
    Test finding of ERA5 files.
    """
    start_time = np.datetime64("2020-01-01T01:15:00")
    end_time = np.datetime64("2020-01-01T01:45:00")

    era5_files = find_era5_sfc_files(ERA5_SFC_PATH, start_time, end_time)
    assert len(era5_files) == 1

    start_time = np.datetime64("2020-01-01T01:15:00")
    end_time = np.datetime64("2020-01-01T02:45:00")
    era5_files = find_era5_sfc_files(ERA5_SFC_PATH, start_time, end_time)
    assert len(era5_files) == 1

    start_time = np.datetime64("2019-12-31T23:45:00")
    end_time = np.datetime64("2020-01-01T00:15:00")
    era5_files = find_era5_sfc_files(ERA5_SFC_PATH, start_time, end_time)
    assert len(era5_files) == 2

    start_time = np.datetime64("2020-01-01T23:45:00")
    end_time = np.datetime64("2020-01-02T00:15:00")
    era5_files = find_era5_sfc_files(ERA5_SFC_PATH, start_time, end_time)
    assert len(era5_files) == 2


@NEEDS_ERA5_DATA
def test_find_era5_lai_files():
    """
    Test finding of ERA5 files.
    """
    start_time = np.datetime64("2020-01-01T01:15:00")
    end_time = np.datetime64("2020-01-01T01:45:00")

    era5_files = find_era5_lai_files(ERA5_LAI_PATH, start_time, end_time)
    assert len(era5_files) == 1

    start_time = np.datetime64("2020-01-01T01:15:00")
    end_time = np.datetime64("2020-01-01T02:45:00")
    era5_files = find_era5_lai_files(ERA5_LAI_PATH, start_time, end_time)
    assert len(era5_files) == 1

    start_time = np.datetime64("2019-12-31T23:45:00")
    end_time = np.datetime64("2020-01-01T00:15:00")
    era5_files = find_era5_lai_files(ERA5_LAI_PATH, start_time, end_time)
    assert len(era5_files) == 2

    start_time = np.datetime64("2020-01-01T23:45:00")
    end_time = np.datetime64("2020-01-02T00:15:00")
    era5_files = find_era5_lai_files(ERA5_LAI_PATH, start_time, end_time)
    assert len(era5_files) == 2


@NEEDS_ERA5_DATA
def test_load_era5_ancillary_data():
    """
    Test loading of ERA5 data.
    """
    start_time = np.datetime64("2020-01-01T01:15:00")
    end_time = np.datetime64("2020-01-01T01:45:00")
    era5_sfc_files = find_era5_sfc_files(ERA5_SFC_PATH, start_time, end_time)
    era5_lai_files = find_era5_lai_files(ERA5_LAI_PATH, start_time, end_time)
    era5_data = load_era5_ancillary_data(era5_sfc_files, era5_lai_files)

    assert len(era5_data.variables) == 12


@NEEDS_INGEST_DATA
def test_find_autosnow_file():
    date = np.datetime64("2020-01-01T01:15:00")
    autosnow_file = find_autosnow_file(INGEST_DIR, date)
    assert autosnow_file.exists()


@NEEDS_INGEST_DATA
def test_load_autosnow_file():
    date = np.datetime64("2020-01-01T01:15:00")
    autosnow_file = find_autosnow_file(INGEST_DIR, date)
    autosnow_data = load_autosnow_data(autosnow_file)


@NEEDS_ANCILLARY_DATA
def test_load_landmask_data():
    ancillary_data = load_landmask_data(ANCILLARY_DIR)


@NEEDS_ANCILLARY_DATA
def test_load_emissivity_data():
    ancillary_data = load_emissivity_data(ANCILLARY_DIR, np.datetime64("2020-01-01T00:00:00"))


@NEEDS_ANCILLARY_DATA
@NEEDS_INGEST_DATA
def test_load_gprof_surface_type_data():
    ancillary_data = load_gprof_surface_type_data(
        ANCILLARY_DIR,
        INGEST_DIR,
        np.datetime64("2020-01-01T00:00:00")
    )
