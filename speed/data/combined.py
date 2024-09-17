"""
spree.data.combined
===================

This module provides functions to extract precipitation retrieval
reference data from GPM CMB and MIRS GMI retrievals.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from tempfile import TemporaryDirectory

from filelock import FileLock
from gprof_nn.data.l1c import L1CFile
import numpy as np
from pansat import Granule
from pansat.catalog import Index
from pansat.products.satellite.gpm import (
    l2b_gpm_cmb,
    l1c_r_gpm_gmi,
    l2a_gpm_dpr_slh
)
from pansat.granule import merge_granules

import xarray as xr

from speed.data.reference import ReferenceData
from speed.data.utils import extract_scans, resample_data
from speed.grids import GLOBAL


LOGGER = logging.getLogger(__name__)


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
    from gprof_nn.data import mirs

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

    def __init__(self, name, include_mirs: bool):
        super().__init__(name, None, l2b_gpm_cmb)
        self.include_mirs = include_mirs

    def load_reference_data(
        self,
        input_granule: Granule,
        granules: List[Granule],
        radius_of_influence: float,
        beam_width: Optional[float]
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

            if self.include_mirs:
                mirs_data = run_mirs(granule, gmi_granules[0])
                gmi_filenames.append(gmi_granules[0].file_record.filename)
                results_mirs.append(mirs_data)

            granule.file_record.get()

            cmb_vars = [
                "latitude",
                "longitude",
                "estim_surf_precip_tot_rate",
                "precipitation_type",
                "precip_tot_water_cont",
                "precip_liq_water_cont",
                "cloud_ice_water_cont",
                "cloud_liq_water_cont",
                "scan_time"
            ]
            cmb_data = granule.open()[cmb_vars].rename(
                {
                    "estim_surf_precip_tot_rate": "surface_precip",
                    "precip_tot_water_cont": "total_water_content",
                    "precip_liq_water_cont": "rain_water_content",
                    "cloud_ice_water_cont": "ice_water_content",
                    "cloud_liq_water_cont": "liquid_water_content"
                })
            profiles = [
                "total_water_content",
                "rain_water_content",
                "ice_water_content",
                "liquid_water_content"
            ]
            for var in profiles:
                var_data = cmb_data[var]
                var_data = var_data.where(var_data > -9000)
                if "vertical_bins" in var_data.dims:
                    var_data = var_data.ffill("vertical_bins")
                var_data.data[:] = var_data.data[..., ::-1]
                cmb_data[var] = var_data
            cmb_data["vertical_bins"] = (("vertical_bins"), 0.125 + 0.25 * np.arange(88))

            swc = cmb_data.total_water_content - cmb_data.rain_water_content
            cmb_data["snow_water_content"] = swc
            cmb_data["snow_water_path"] = swc.integrate("vertical_bins")
            cmb_data["rain_water_path"] = cmb_data["rain_water_content"].integrate("vertical_bins")
            cmb_data["ice_water_path"] = cmb_data["ice_water_content"].integrate("vertical_bins")
            cmb_data["liquid_water_path"] = cmb_data["liquid_water_content"].integrate("vertical_bins")

            cmb_type = cmb_data["precipitation_type"].data.astype("int64")
            precip_type = np.zeros_like(cmb_type, dtype="uint")
            precip_type[cmb_type == -1111] = 1
            precip_type[cmb_type // 10_000_000 == 1] = 2
            precip_type[cmb_type // 10_000_000 == 2] = 3
            precip_type[cmb_type // 10_000_000 == 3] = 4

            cmb_data["precipitation_type"] = (
                list(cmb_data["precipitation_type"].dims),
                precip_type
            )
            target_levels = np.concatenate([0.25 * np.arange(20) + 0.25, 10.5 + np.arange(8)])
            cmb_data = cmb_data.interp(vertical_bins=target_levels)

            lock = FileLock("slh.lock")
            with lock:
                start_time = cmb_data["scan_time"].data[0]
                end_time = cmb_data["scan_time"].data[-1]
                central_time = start_time + 0.5 * (end_time - start_time)
                slh_recs = l2a_gpm_dpr_slh.get(central_time)
                if len(slh_recs) > 0:
                    if len(slh_recs) > 1:
                        LOGGER.error(
                            "Found more than 1 SLH record!"
                        )
                        return None
                    scan_start, scan_end = granule.primary_index_range
                    slh_data = l2a_gpm_dpr_slh.open(slh_recs[0])[{"scans": slice(scan_start, scan_end)}]
                    slh_data = slh_data[["latent_heating"]].rename({
                        "scans": "matched_scans",
                        "pixels": "matched_pixels"
                    })
                    slh_data = slh_data.assign_coords(matched_scans=cmb_data.matched_scans.data)
                    mask = slh_data.latent_heating.data < -9000
                    slh_data["latent_heating"].data[mask] = np.nan
                    if np.all(np.isnan(slh_data["latent_heating"].data)):
                        LOGGER.warning(
                            "No valid latent heating rates in SLH data."
                        )
                    slh_data["vertical_bins"] = (("vertical_bins"), 0.125 + 0.25 * np.arange(80))
                    slh = slh_data.latent_heating.interp(vertical_bins=target_levels)
                else:
                    LOGGER.warning(
                        "Found no SLH records for %s.", granule.time_range
                    )
                    slh = xr.DataArray(np.zeros_like(swc.data), dims=("scans", "pixels", "vertical_bins"))

            cmb_data["latent_heating"] = slh

            results_cmb.append(cmb_data)
            cmb_filenames.append(granule.file_record.filename)

        cmb_data = xr.concat(results_cmb, "matched_scans")

        lons, lats = GLOBAL.grid.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]

        lon_min = cmb_data.longitude.data.min()
        lon_max = cmb_data.longitude.data.max()
        shifted = False
        if lon_max - lon_min > 180:
            shifted = True
            lons_cmb = cmb_data.longitude.data
            lons_cmb = (lons_cmb % 360) - 180
            lon_min = lons_cmb.min()
            lon_max = lons_cmb.max()

        cols = np.where((lon_min <= lons) * (lon_max >= lons))[0]
        col_start = cols.min()
        col_end = cols.max()
        assert col_end - col_start < 3500

        lat_min = cmb_data.latitude.data.min()
        lat_max = cmb_data.latitude.data.max()
        rows = np.where((lat_min <= lats) * (lat_max >= lats))[0]
        row_start = rows.min()
        row_end = rows.max()

        LOGGER.info(
            "Restricting grid %s %s %s %s",
            col_start, col_end, row_start, row_end
        )

        grid = GLOBAL.grid[row_start: row_end, col_start:col_end]
        cmb_data["time"], _ = xr.broadcast(cmb_data.scan_time, cmb_data.latitude)

        data_r = resample_data(cmb_data, grid, 5e3)
        data_r.attrs["input_files"] = ",".join(cmb_filenames)

        if self.include_mirs:

            mirs_data = xr.concat(results_mirs, "Scanline")
            if shifted:
                mirs_data.longitude.data = (
                    (mirs_data.longitude.data % 360) - 180
                )
            mirs_data_r = resample_data(mirs_data, GLOBAL.grid, 15e3)
            data_r = xr.merge([mirs_data_r, data_r])
            data_r.attrs["gmi_l1c_files"] = ",".join(str(gmi_filenames))

        if shifted:
            lons = data_r.longitude.data
            lons[:] = (lons + 360) % 360 - 180

        data_r.attrs["lower_left_col"] = col_start
        data_r.attrs["lower_left_row"] = row_start

        float_encoding = {"dtype": "float32", "compression": "zstd"}
        int_encoding = {"dtype": "int8", "compression": "zstd"}

        float_vars = [
            "surface_precip",
            "total_water_content",
            "rain_water_content",
            "ice_water_content",
            "liquid_water_content",
            "snow_water_path",
            "rain_water_path",
            "ice_water_path",
            "liquid_water_path",
            "latent_heating"
        ]
        for var in float_vars:
            data_r[var].encoding = float_encoding
        data_r["precipitation_type"].encoding = int_encoding

        if beam_width is None:
            return data_r, None

        raise RuntimeError(
            "Calculation of footprint-averaged results not yet supported."
        )


gpm_cmb = Combined("cmb", include_mirs=False)
gpm_cmb_w_mirs = Combined("cmb_w_mirs", include_mirs=True)
