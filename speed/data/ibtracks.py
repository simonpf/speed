from typing import List, Optional


import numpy as np
from pansat.granule import Granule
from pansat.products.reanalysis import ibtracks as ibt
from pansat.utils import resample_data
from pyresample.geometry import SwathDefinition
import xarray as xr

from speed.grids import GLOBAL
from speed.data.reference import ReferenceData
from speed.data.utils import calculate_footprint_averages


class IBTracks(ReferenceData):
    """
    Reference data class for extracting OceanRAIN data.
    """

    def __init__(self):
        super().__init__(
            "ibtracks",
            None,
            ibt.ibtracks
        )

    def load_reference_data(
        self,
        input_granule: Granule,
        granules: List[Granule],
        radius_of_influence: float,
        beam_width: float
    ) -> Optional[xr.Dataset]:
        """
        Load reference data for a given granule of input data.

        Args:
            input_granule: The matched input data granules.
            granules: A list containing the matched reference data granules.
            radius_of_influence: The radius of influence to use for the resampling
                of the scan times.
            beam_width: The beam width to use for the calculation of footprint-averaged
                precipitation.

        Return:
            An xarray.Dataset containing the MRMS reference data.
        """
        data = granules[0].open()[{"storm": 0}]
        lons = data.lon.data
        lats = data.lat.data
        swath = SwathDefinition(lons=lons, lats=lats)

        idx_range = input_granule.primary_index_range
        idx_range = (idx_range[0] - 64, idx_range[-1] + 64)
        input_granule.primary_index_range = idx_range
        input_data = input_granule.open().reset_coords()
        input_time, _ = xr.broadcast(input_data.scan_time, input_data.longitude_s1)
        input_data["scan_time"] = input_time
        input_data["latitude"] = (("scans", "pixels",), input_data["latitude_s1"].data)
        input_data["longitude"] = (("scans", "pixels"), input_data["longitude_s1"].data)
        resample_vars = ["scan_time", "latitude", "longitude", "latitude_s1", "longitude_s1"]
        input_data_r = resample_data(
            input_data[resample_vars],
            swath,
            radius_of_influence=60e3,
            new_dims=("time",)
        )

        time_diff = np.abs(input_data_r.scan_time.data - data.time.data)
        valid = (time_diff < np.timedelta64(180, "m"))

        start_time = data[{"date_time": valid}].time.min().data - np.timedelta64(90, "m")
        end_time = data[{"date_time": valid}].time.max().data + np.timedelta64(90, "m")
        print(start_time, end_time, input_data.scan_time)
        scan_mask = ((start_time <= input_data.scan_time) * (input_data.scan_time <= end_time)).any(dim="pixels_s1")
        input_data = input_data[{"scans": scan_mask}]
        lons = input_data.longitude_s1.data
        lats = input_data.latitude_s1.data
        lon_min = np.nanmin(lons) - 1.0
        lon_max = np.nanmax(lons) + 1.0
        lat_min = np.nanmin(lats) - 1.0
        lat_max = np.nanmax(lats) + 1.0

        print("BNDS :: ", lon_min, lat_min, lon_max, lat_max)
        print(input_data_r.latitude_s1)
        print(input_data_r.longitude_s1)

        if 180 < lon_max - lon_min:
            lon_min = 0.0
            lon_max = 0.0
        swath = SwathDefinition(lons=lons, lats=lats)

        # Get global grid
        lons_g = GLOBAL.lons.copy()
        lats_g = GLOBAL.lats.copy()
        lons_g = lons_g
        lats_g = lats_g
        lon_inds = (lon_min < lons_g) * (lons_g < lon_max)
        lons_g = lons_g[lon_inds]
        lat_inds = (lat_min < lats_g) * (lats_g < lat_max)
        lats_g = lats_g[lat_inds]
        lower_left_col = np.where(lon_inds)[0][0]
        lower_left_row = np.where(lat_inds)[0][0]
        lon_bins = 0.5 * (lons_g[1:] + lons_g[:-1])
        lat_bins = 0.5 * (lats_g[1:] + lats_g[:-1])

        time = np.zeros((lats_g.size, lons_g.size), dtype="datetime64[s]")
        time[:] = input_data.scan_time.mean().data
        results = xr.Dataset({
            "longitude": (("longitude",), lons_g),
            "latitude": (("latitude",), lats_g),
            "surface_precip": (("latitude", "longitude"), np.zeros((lats_g.size, lons_g.size))),
            "time": (("latitude", "longitude",), time),
        })
        results.attrs["lower_left_row"] = lower_left_row + 1
        results.attrs["lower_left_col"] = lower_left_col + 1
        if beam_width is None:
            return results, None

        results_fpavg = calculate_footprint_averages(
            data,
            lons_fp,
            lats_fp,
            sensor_longitude,
            sensor_latitude,
            sensor_altitude,
            beam_width,
            0.5
        )
        return results, results_fpavg


ibtracks = IBTracks()
