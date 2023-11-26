from datetime import datetime

import pytest

from pansat.time import TimeRange
from pansat.products.satellite.gpm import (
    l1c_metopb_mhs,
)
from pansat.catalog import Index
from pansat.environment import get_index
from pansat.catalog.index import find_matches
from pansat.products.ground_based import mrms
from pansat.products.satellite.gpm import (
    l1c_r_gpm_gmi,
    l2b_gpm_cmb
)

from speed.data.gpm import run_preprocessor


@pytest.fixture
def gpm_mrms_files(scope="session"):
    start_time = "2020-01-01T00:00:00"
    end_time = "2020-01-02T00:00:00"
    time_range = TimeRange(start_time, end_time)


@pytest.fixture
def gmi_granule(scope="session"):
    start_time = "2020-01-01T00:00:00"
    end_time = "2020-01-01T12:00:00"
    time_range = TimeRange(start_time, end_time)
    gmi_files = l1c_r_gpm_gmi.find_files(time_range, roi=mrms.MRMS_DOMAIN)
    gmi_files = [gmi_files[0].get()]
    index = Index.index(l1c_r_gpm_gmi, gmi_files)
    granules = index.find(roi=mrms.MRMS_DOMAIN)
    return granules[0]


@pytest.fixture
def gmi_granule(scope="session"):
    start_time = "2020-01-01T00:00:00"
    end_time = "2020-01-01T12:00:00"
    time_range = TimeRange(start_time, end_time)
    gmi_files = l1c_r_gpm_gmi.find_files(time_range, roi=mrms.MRMS_DOMAIN)
    gmi_files = [gmi_files[0].get()]
    index = Index.index(l1c_r_gpm_gmi, gmi_files)
    granules = index.find(roi=mrms.MRMS_DOMAIN)
    return granules[0]


@pytest.fixture
def cmb_granule(scope="session"):
    start_time = "2020-01-01T00:00:00"
    end_time = "2020-01-01T12:00:00"
    time_range = TimeRange(start_time, end_time)
    cmb_files = l2b_gpm_cmb.find_files(time_range, roi=mrms.MRMS_DOMAIN)
    cmb_files = [cmb_files[0].get()]
    index = Index.index(l2b_gpm_cmb, cmb_files)
    granules = index.find(roi=mrms.MRMS_DOMAIN)
    return granules[0]


@pytest.fixture
def cmb_match(scope="session"):
    """
    Returns a tuple describing a matche between a GMI and a GPM CMB granule.
    """
    start_time = "2020-01-01T00:12:00"
    end_time = "2020-01-02T00:12:00"
    time_range = TimeRange(start_time, end_time)
    cmb_index = get_index(l2b_gpm_cmb).subset(time_range)
    gmi_index = get_index(l1c_r_gpm_gmi).subset(time_range)
    matches = find_matches(gmi_index, cmb_index)
    return matches[0]


@pytest.fixture
def mrms_match(scope="session"):
    """
    Returns a tuple describing a matche between a GMI and a GPM CMB granule.
    """
    start_time = "2020-01-01T00:12:00"
    end_time = "2020-01-02T00:12:00"
    time_range = TimeRange(start_time, end_time)
    cmb_index = get_index(mrms.precip_rate).subset(time_range)
    gmi_index = get_index(l1c_r_gpm_gmi).subset(time_range)
    matches = find_matches(gmi_index, cmb_index)
    return matches[0]


@pytest.fixture(scope="session")
def mhs_conus_granule():
    gpm_prod = l1c_metopb_mhs
    time_range = TimeRange("2019-01-01T00:00:00", "2019-01-02T00:00:00")
    gpm_recs = gpm_prod.get(time_range, roi=mrms.MRMS_DOMAIN)
    index = Index.index(gpm_prod, gpm_recs[:1])
    granules = index.find(roi=mrms.MRMS_DOMAIN)
    assert len(granules) > 1
    return granules[0]


@pytest.fixture(scope="session")
def preprocessor_data(mhs_conus_granule):
     preprocessor_data = run_preprocessor(mhs_conus_granule)
     return preprocessor_data
