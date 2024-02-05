"""
spree.data.combined
===================

This module provides functions to extract precipitation retrieval
reference data from GPM CMB and MIRS GMI retrievals.
"""
import os
from pathlib import Path
from typing import List, Optional
from tempfile import TemporaryDirectory

from gprof_nn.data import mirs
from gprof_nn.data.l1c import L1CFile
import numpy as np
from pansat import Granule
from pansat.catalog import Index
from pansat.products.satellite.gpm import l2b_gpm_cmb, l1c_r_gpm_gmi
from pansat.granule import merge_granules

import xarray as xr

from speed.data.reference import ReferenceData
from speed.data.utils import extract_scans, resample_data
from speed.grids import GLOBAL


def run_mirs(cmb_granule, gmi_granule):
    """
    Run MIRS retrieval on a GPM GMI granule.

    Args:
        cmb_granule: The GPM CMB granule that the MIRS retrievals will
            be mapped to.
        gmi_granule: A pansat granule identifying a subset of an orbit
            of GPM GMI L1C file.

    Return:
        An xarray.Dataset containing the results from the mirs retrieval.
    """

    old_dir = os.getcwd()
    try:
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            input_dir = tmp / "input"
            input_dir.mkdir()
            process_dir = tmp / "process"
            process_dir.mkdir()

            l1c_file = extract_scans(gmi_granule, input_dir)
            assert l1c_file.exists()
            os.chdir(process_dir)

            config = mirs.MIRSConfig(Path("/xdata/simon/mirs/mirs_dap"), process_dir)
            mirs_results = mirs.run_mirs_retrieval(config, l1c_file, tmp)
    finally:
        os.chdir(old_dir)

    # Keep only subset of results
    mirs_results = mirs_results[1].rename(
        {
            "RR": "surface_precip_mirs",
            "ChiSqr": "chi_squared_mirs",
            "Qc": "quality_flag_mirs",
            "Longitude": "longitude",
            "Latitude": "latitude",
        }
    )
    vars = [
        "surface_precip_mirs",
        "chi_squared_mirs",
        "quality_flag_mirs",
        "latitude",
        "longitude",
    ]
    return mirs_results[vars]


class Combined(ReferenceData):
    """
    Reference data class for processing GPM CMB data.

    Extracts surface precipitation from GPM CMB and MIRS.
    """

    def __init__(self, name):
        super().__init__(name, None, l2b_gpm_cmb)

    def load_reference_data(
        self, input_granule: Granule, granules: List[Granule]
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of GPM CMB data.

        Args:
            input_granule: Granule of the input data matching the reference
                data.
            granules: A list of granule objects specifying the reference
                data to load.

        Return:
            An xarray.Dataset containing the reference data extracted frmo
            GPM CMB retrieval results and collocated MIRS measurements.
        """
        results_mirs = []
        results_cmb = []

        gmi_filenames = []
        cmb_filenames = []

        for granule in granules:
            gmi_files = l1c_r_gpm_gmi.get(time_range=granule.time_range)
            index = Index.index(l1c_r_gpm_gmi, gmi_files)
            gmi_granules = merge_granules(
                index.find(time_range=granule.time_range, roi=granule.geometry)
            )
            mirs_data = run_mirs(granule, gmi_granules[0])
            gmi_filenames.append(gmi_granules[0].file_record.filename)
            results_mirs.append(mirs_data)

            granule.file_record.get()

            cmb_vars = ["latitude", "longitude", "estim_surf_precip_tot_rate"]
            cmb_data = granule.open()[cmb_vars].rename(
                {"estim_surf_precip_tot_rate": "surface_precip"}
            )
            results_cmb.append(cmb_data)
            cmb_filenames.append(granule.file_record.filename)

        mirs_data = xr.concat(results_mirs, "Scanline")
        cmb_data = xr.concat(results_cmb, "matched_scans")

        mirs_data_r = resample_data(mirs_data, GLOBAL.grid, 15e3)
        cmb_data_r = resample_data(cmb_data, GLOBAL.grid, 5e3)

        data_r = xr.merge([mirs_data_r, cmb_data_r])
        data_r.attrs["gmi_l1c_files"] = ",".join(str(gmi_filenames))
        data_r.attrs["cmb_files"] = ",".join(str(cmb_filenames))
        return data_r


gpm_cmb = Combined("cmb")
