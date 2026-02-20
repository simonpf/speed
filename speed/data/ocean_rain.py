"""
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
from pansat.utils import resample_data
from scipy.stats import binned_statistic_2d
from pyresample.geometry import SwathDefinition

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
                idx_range = granule.primary_index_range
                idx_range = (max(idx_range[0] - 120, 0), idx_range[1] + 120)
                granule.primary_index_range = idx_range
                granule_data = granule.open()
                data.append(granule_data)
            except ValueError:
                continue

        data = xr.concat(data, dim="time").rename(
         ODM470_precipitation_rate_R="surface_precip"
        )
        lons = data.longitude.data
        lats = data.latitude.data
        lon_min = np.nanmin(lons) - 0.1
        lon_max = np.nanmax(lons) + 0.1
        lat_min = np.nanmin(lats) - 0.1
        lat_max = np.nanmax(lats) + 0.1
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

        input_data = input_granule.open().reset_coords()
        input_time, _ = xr.broadcast(input_data.scan_time, input_data.longitude_s1)
        input_data["scan_time"] = input_time
        input_data["latitude"] = (("scans", "pixels",), input_data["latitude_s1"].data)
        input_data["longitude"] = (("scans", "pixels"), input_data["longitude_s1"].data)
        resample_vars = ["scan_time", "latitude", "longitude", "latitude_s1", "longitude_s1"]
        input_data_r = resample_data(input_data[resample_vars], swath, radius_of_influence=7.5e3, new_dims=("time",))

        time_diff = np.abs(input_data_r.scan_time.data - data.time.data)

        results = []

        for delta in [15, 30, 60]:
            valid = (time_diff < np.timedelta64(delta, "m")) * (np.isfinite(data.surface_precip.data))
            weights = binned_statistic_2d(
                input_data_r.latitude_s1.data[valid],
                input_data_r.longitude_s1.data[valid],
                data.surface_precip.data[valid],
                statistic="count",
                bins=(lat_bins[::-1], lon_bins)
            )[0][::-1]
            weights /= weights.sum()
            mask = 0.0 < weights

            interval = 2 * delta
            nearest_time = np.argmin(time_diff)
            surface_precip = data.surface_precip.rolling(time=interval, center=True).mean()[{"time": nearest_time}].data
            surface_precip_std = data.surface_precip.rolling(time=interval, center=True).std()[{"time": nearest_time}].data
            p_rain = data.probability_for_rain.rolling(time=interval, center=True).mean()[{"time": nearest_time}].data
            p_snow = data.probability_for_snow.rolling(time=interval, center=True).mean()[{"time": nearest_time}].data
            p_mixed = data.probability_for_mixed_phase.rolling(time=interval, center=True).mean()[{"time": nearest_time}].data
            surface_precip_gauge = data.rain_gauge_precipitation_rate.rolling(time=interval, center=True).mean()[{"time": nearest_time}].data

            field = np.nan * np.ones_like(mask)
            field[mask] = 1.0

            results.append(xr.Dataset({
                f"weights_{interval}": (("latitude", "longitude"), weights),
                f"surface_precip_{interval}": (("latitude", "longitude"), field * surface_precip),
                f"surface_precip_std_{interval}": (("latitude", "longitude"), field * surface_precip_std),
                f"surface_precip_gauge_{interval}": (("latitude", "longitude"), field * surface_precip_gauge),
                f"probability_of_rain_{interval}": (("latitude", "longitude"), field * p_rain),
                f"probability_of_snow_{interval}": (("latitude", "longitude"), field * p_snow),
                f"probability_of_mixed_{interval}": (("latitude", "longitude"), field * p_mixed),
            }))

        results = xr.merge(results)
        results["surface_precip"] = results.surface_precip_60
        results["longitude"] = (("longitude",), 0.5 * (lon_bins[1:] + lon_bins[:-1]))
        results["latitude"] = (("latitude",), 0.5 * (lat_bins[1:] + lat_bins[:-1]))
        results.attrs["lower_left_row"] = lower_left_row + 1
        results.attrs["lower_left_col"] = lower_left_col + 1

        time = data.time[{"time": nearest_time}].data
        time_field = np.zeros_like(mask, dtype=time.dtype)
        time_field[:] = time
        results["time"] = (("latitude", "longitude"), time_field)

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
    ocean_rain_rv_investigator
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
