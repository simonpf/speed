"""
speed.data.wegener_net
======================

This module provides functionality to extract collocations with the WegenerNet
ground-station data.
"""
import logging
from typing import List, Optional

import numpy as np
from pansat.products.stations.wegener_net import station_data, get_station_data
from pansat.geometry import LonLatRect
from pansat.granule import Granule
from scipy.stats import binned_statistic_2d

from speed.data.reference import ReferenceData
from speed.data.utils import calculate_footprint_averages
from speed.grids import GLOBAL

import xarray as xr


def get_domain() -> LonLatRect:
    """
    Return a lat/lon bounding box defining the spatial coverage of the WegenerNet data.
    """
    station_data = get_station_data()
    lons = station_data.longitude.data
    lats = station_data.latitude.data
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    return LonLatRect(lon_min, lat_min, lon_max, lat_max)


LOGGER = logging.getLogger(__name__)


class WegenerNet(ReferenceData):
    """
    Reference data class for processing MRMS data.

    Combines MRMS precip rate, flag and radar quality index into
    a DataArray.
    """

    def __init__(self):
        super().__init__("wegener_net", get_domain(), station_data)

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
        coords = input_granule.geometry.bounding_box_corners
        lon_min, lat_min, lon_max, lat_max = coords

        wegener_data = []
        for granule in granules:
            try:
                station_data = granule.open()
                if "latitude" in station_data.dims:
                    continue
                wegener_data.append(station_data)
            except ValueError:
                continue
        wegener_data = xr.concat(wegener_data, dim="station")
        invalid = 0.0 < wegener_data.flagged_percentage
        wegener_data.surface_precip.data[invalid] = np.nan

        lons_g = GLOBAL.lons.copy()
        lats_g = GLOBAL.lats.copy()
        lons_g = lons_g
        lats_g = lats_g

        lon_inds = (lons_g > lon_min) * (lons_g < lon_max)
        lons_g = lons_g[lon_inds]
        lat_inds = (lats_g > lat_min) * (lats_g < lat_max)
        lats_g = lats_g[lat_inds]
        lower_left_col = np.where(lon_inds)[0][0]
        lower_left_row = np.where(lat_inds)[0][0]

        lon_bins = 0.5 * (lons_g[1:] + lons_g[:-1])
        lat_bins = 0.5 * (lats_g[1:] + lats_g[:-1])

        input_data = input_granule.open()
        scan_time = input_data.scan_time.data
        start_time = np.nanmin(scan_time)
        end_time = np.nanmax(scan_time)
        time = start_time + 0.5 * (end_time - start_time)

        wegener_data = wegener_data.interp(time=time, method="nearest")

        surface_precip = binned_statistic_2d(
            wegener_data.latitude.data,
            wegener_data.longitude.data,
            2.0 * wegener_data.surface_precip.data, # This is half-hourly data.
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]

        lons = lons_g[1:-1]
        lats = lats_g[1:-1]

        time = np.broadcast_to(time[None, None], (lats.size, lons.size))

        results = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("latitude", "longitude"), surface_precip),
            "time": (("latitude", "longitude"), time)
        })
        # Account for shrinkage caused by grid-based resampling
        results.attrs["lower_left_row"] = lower_left_row + 1
        results.attrs["lower_left_col"] = lower_left_col + 1

        input_data = input_granule.open()
        lats_fp = input_data.latitude if "latitude" in input_data else input_data.latitude_s1
        lons_fp = input_data.longitude if "longitude" in input_data else input_data.longitude_s1
        sensor_latitude = input_data.spacecraft_latitude
        sensor_longitude = input_data.spacecraft_longitude
        sensor_altitude = input_data.spacecraft_altitude

        if beam_width is None:
            return results, None

        results_fpavg = calculate_footprint_averages(
            wegener_data,
            lons_fp,
            lats_fp,
            sensor_longitude,
            sensor_latitude,
            sensor_altitude,
            beam_width,
            0.5
        )


        return results, results_fpavg


wegener_net = WegenerNet()
