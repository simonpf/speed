"""gpmn
speed.data.wegener_net
======================

This module provides functionality to extract collocations with the OceanRAIN
ship-based precipitation measurements.
"""
import logging
from typing import List, Optional

import numpy as np
from pansat.products.ground_based.ocean_rain import (
    ocean_rain_ms_the_world,
    ocean_rain_rv_investigator,
    ocean_rain_rv_maria_s_merian,
    ocean_rain_rv_meteor,
    ocean_rain_rv_polarstern,
    ocean_rain_rv_roger_revelle,
    ocean_rain_rv_sonneii,
)

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
    return LonLatRect(-180, -90, 180, 90)


LOGGER = logging.getLogger(__name__)


class OceanRain(ReferenceData):
    """
    Reference data class for extracting OceanRAIN data.

    Extracts matching
    a DataArray.
    """

    def __init__(
            self,
            name: str,
            pansat_product: "pansat.Product"
    ):
        super().__init__(
            name,
            None,
            pansat_product
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
        data = []
        for granule in granules:
            try:
                granule_data = granule.open()
                data.append(granule_data)
            except ValueError:
                continue
        data = xr.concat(data, dim="time").rolling(time=30)
        invalid = 3 < data.precip_flag
        data.surface_precip.data[invalid] = np.nan

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

        data = data.interp(time=time, method="nearest")

        rain_fraciton = binned_statistic_2d(
            data.latitude.data,
            data.longitude.data,
            data.precip_flag.data == 0,
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]
        snow_fraction = binned_statistic_2d(
            data.latitude.data,
            data.longitude.data,
            data.precip_flag.data == 1,
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]
        mixed_phase_fraction = binned_statistic_2d(
            data.latitude.data,
            data.longitude.data,
            data.precip_flag.data == 2,
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]
        precip_fraction = binned_statistic_2d(
            data.latitude.data,
            data.longitude.data,
            data.precip_flag.data < 3,
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]

        surface_precip = binned_statistic_2d(
            data.latitude.data,
            data.longitude.data,
            data.ODM470_precipitation_rate_R.data == 0.0,
            bins=(lat_bins[::-1], lon_bins)
        )[0][::-1]

        lons = lons_g[1:-1]
        lats = lats_g[1:-1]

        time = np.broadcast_to(time[None, None], (lats.size, lons.size))

        results = xr.Dataset({
            "longitude": (("longitude",), lons),
            "latitude": (("latitude",), lats),
            "surface_precip": (("latitude", "longitude"), surface_precip),
            "precip_fraction": (("latitude", "longitude"), precip_fraction),
            "rain_fraction": (("latitude", "longitude"), rain_fraction),
            "snow_fraction": (("latitude", "longitude"), snow_fraction),
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


ocean_rain_ms_world = OceanRain(
    "ocean_rain_ms_the_world",
    ocean_rain_ms_the_world
)
ocean_rain_rv_maria_s_merian = OceanRain(
    "ocean_rain_rv_maria_s_merian",
    ocean_rain_rv_maria_s_merian
)
ocean_rain_rv_investigator = OceanRain(
    "ocean_rain_rv_investigator",
    ocean_rain_rv_maria_s_merian
)
ocean_rain_rv_meteor = OceanRain(
    "ocean_rain_rv_meteor",
    ocean_rain_rv_meteor
)
ocean_rain_rv_meteor = OceanRain(
    "ocean_rain_rv_polarstern",
    ocean_rain_rv_polarstern
)
ocean_rain_rv_roger_revelle = OceanRain(
    "ocean_rain_rv_roger_revelle",
    ocean_rain_rv_roger_revelle
)
ocean_rain_rv_polarstern = OceanRain(
    "ocean_rain_rv_polarstern",
    ocean_rain_rv_polarstern
)
ocean_rain_rv_sonneii = OceanRain(
    "ocean_rain_rv_sonneii",
    ocean_rain_rv_sonneii
)
