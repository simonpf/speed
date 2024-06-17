"""
speed.plotting
==============

Provides convenience function for plotting collocations.
"""
from pathlib import Path
from typing import Optional


import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FixedLocator
import numpy as np
import xarray as xr


from speed import grids


def add_ticks(ax, lons, lats, left=True, bottom=True):
    """
    Add ticks to plot.

    Args:
        ax: A matplotib axes object to which to add axes.
        lons: The longitude positions of the ticks along the x-axis.
        lats: The latitude postitions of the ticks along the y-axis.
        left: Whether to draw ticks on the y-axis.
        bottom: Wheter tod draw ticks on the x-axis.
    """
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color="none"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left
    gl.bottom_labels = bottom
    gl.xlocator = FixedLocator(lons)
    gl.ylocator = FixedLocator(lats)


def plot_collocations_on_swath(
    collocation_file: Path,
    axs: Optional[np.ndarray] = None,
    tb_cmap: str = "cmo.thermal",
    precip_cmap: str = "cmo.dense_r",
    draw_x_ticks: bool = True,
    draw_y_ticks: bool = True,
    add_titles: bool = True,
    channel: int = 0,
) -> np.ndarray:
    """
    Plot collocations in on_swath format.

    Args:
        collocation_file: Path object pointing to location of SPEED
            collocation file.
        axs: Optional 1D array of matplotlib axes object to plot the data
            into.
        cmap: Optional name of a colormap to use for plotting.

    Return:
        numpy.ndarray containing the axes that the data has been plotted
        to.
    """
    precip_norm = LogNorm(1e-1, 1e2)
    tb_norm = Normalize(150, 300)

    if axs is None:
        crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 5, width_ratios=[0.1, 1.0, 1.0, 1.0, 0.1], hspace=0.25)

        axs = np.array(
            [
                fig.add_subplot(gs[0, i], projection=None if i in [0, 4] else crs)
                for i in range(5)
            ]
        )

    data_inpt = xr.load_dataset(collocation_file, group="input_data")
    data_ref = xr.load_dataset(collocation_file, group="reference_data")

    sensor = data_inpt.attrs["sensor"]
    satellite = data_inpt.attrs["satellite"]
    freqs = data_inpt.attrs["frequencies"]

    if len(axs) < 5:
        axs = np.concatenate([[None], axs, [None]], 0)

    tbs_mw = data_inpt.tbs_mw.data
    tbs_ir = data_inpt.tbs_ir.data

    if "surface_precip" in data_ref:
        surface_precip = np.maximum(data_ref.surface_precip.data, 1e-3)
    else:
        surface_precip = np.maximum(data_ref.surface_precip_combined.data, 1e-3)

    lons = data_inpt.longitude.data
    lon_c = 0.5 * (lons.max() + lons.min())
    lats = data_inpt.latitude.data
    lat_c = 0.5 * (lats.max() + lats.min())
    d_lon = lons.max() - lons.min()
    d_lat = lats.max() - lats.min()
    ext = max(d_lon, d_lat)
    lon_min = lon_c - 0.5 * ext
    lon_max = lon_c + 0.5 * ext
    lat_min = lat_c - 0.5 * ext
    lat_max = lat_c + 0.5 * ext
    tick_res = 10
    x_ticks = np.arange(
        (lon_min // tick_res) * tick_res,
        ((lon_max // tick_res) + 1) * tick_res + 1e-6,
        tick_res,
    )
    y_ticks = np.arange(
        (lat_min // tick_res) * tick_res,
        ((lat_max // tick_res) + 1) * tick_res + 1e-6,
        tick_res,
    )

    ax = axs[1]
    ax.pcolormesh(lons, lats, tbs_mw[..., channel], norm=tb_norm, cmap=tb_cmap)
    ax.coastlines(color="grey")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_ticks(ax, x_ticks, y_ticks, left=draw_y_ticks, bottom=draw_x_ticks)

    if add_titles:
        ax.set_title(f"{satellite} {sensor} ({freqs[0]:3.2f} GHz)")

    ax = axs[2]
    m_tbs = ax.pcolormesh(lons, lats, tbs_ir[...], norm=tb_norm, cmap=tb_cmap)
    ax.coastlines(color="grey")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_ticks(ax, x_ticks, y_ticks, left=False, bottom=draw_x_ticks)

    if add_titles:
        ax.set_title(f"Geo IR")

    ax = axs[3]
    m_sp = ax.pcolormesh(lons, lats, surface_precip, norm=precip_norm, cmap=precip_cmap)
    ax.coastlines(color="grey")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_ticks(ax, x_ticks, y_ticks, left=False, bottom=draw_x_ticks)

    if add_titles:
        ax.set_title(f"Surface precipitation")
    #
    # Add colorbars
    #

    if axs[0] is not None:
        plt.colorbar(m_tbs, label="Brightness temperatures [K]", cax=axs[0])
        axs[0].yaxis.set_ticks_position("left")
        axs[0].yaxis.set_label_position("left")

    if axs[-1] is not None:
        plt.colorbar(m_sp, label="Surface precipitation [mm h$^{-1}$]", cax=axs[-1])

    return axs


def plot_collocations_gridded(
    collocation_file: Path,
    axs: Optional[np.ndarray] = None,
    tb_cmap: str = "cmo.thermal",
    precip_cmap: str = "cmo.dense_r",
    draw_x_ticks: bool = True,
    draw_y_ticks: bool = True,
    channel: int = 0,
) -> np.ndarray:
    """
    Plot collocations in gridded format.

    Args:
        collocation_file: path object pointing to location of speed
            collocation file.
        axs: optional 1d array of matplotlib axes object to plot the data
            into.
        cmap: optional name of a colormap to use for plotting.

    Return:
        numpy.ndarray containing the axes that the data has been plotted
        to.
    """
    precip_norm = LogNorm(1e-1, 1e2)
    tb_norm = Normalize(150, 300)

    data_inpt = xr.load_dataset(collocation_file, group="input_data")
    data_ref = xr.load_dataset(collocation_file, group="reference_data")

    if axs is None:
        crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 5, width_ratios=[0.1, 1.0, 1.0, 1.0, 0.1], wspace=0.3)

        axs = np.array(
            [
                fig.add_subplot(gs[0, i], projection=None if i in [0, 4] else crs)
                for i in range(5)
            ]
        )

    ll_row = data_inpt.lower_left_row
    ll_col = data_inpt.lower_left_col
    n_rows = data_inpt.dims["latitude"]
    n_cols = data_inpt.dims["longitude"]
    grid = grids.GLOBAL.grid[
        slice(ll_row, ll_row + n_rows), slice(ll_col, ll_col + n_cols)
    ]

    lons = data_ref.longitude.data
    lon_c = 0.5 * (lons.max() + lons.min())
    lats = data_ref.latitude.data
    lat_c = 0.5 * (lats.max() + lats.min())
    d_lon = lons.max() - lons.min()
    d_lat = lats.max() - lats.min()
    ext = max(d_lon, d_lat)
    lon_min = lon_c - 0.5 * ext
    lon_max = lon_c + 0.5 * ext
    lat_min = lat_c - 0.5 * ext
    lat_max = lat_c + 0.5 * ext
    tick_res = 10
    x_ticks = np.arange(
        (lon_min // tick_res) * tick_res,
        ((lon_max // tick_res) + 1) * tick_res + 1e-6,
        tick_res,
    )
    y_ticks = np.arange(
        (lat_min // tick_res) * tick_res,
        ((lat_max // tick_res) + 1) * tick_res + 1e-6,
        tick_res,
    )

    ext = grid.area_extent
    ext = (ext[0], ext[2], ext[1], ext[3])

    if len(axs) < 5:
        axs = np.concatenate([[none], axs, [none]], 0)

    tbs_mw = data_inpt.tbs_mw.data
    tbs_ir = data_inpt.tbs_ir.data
    if "surface_precip" in data_ref:
        surface_precip = np.maximum(data_ref.surface_precip.data, 1e-3)
    else:
        surface_precip = np.maximum(data_ref.surface_precip_combined.data, 1e-3)
    lons = data_inpt.longitude.data
    lats = data_inpt.latitude.data

    ax = axs[1]
    ax.imshow(
        tbs_mw[..., channel], norm=tb_norm, cmap=tb_cmap, extent=ext, origin="upper"
    )
    ax.coastlines(color="grey")
    add_ticks(ax, x_ticks, y_ticks, left=draw_y_ticks, bottom=draw_x_ticks)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax = axs[2]
    m_tbs = ax.imshow(
        tbs_ir, norm=tb_norm, cmap=tb_cmap, extent=ext, origin="upper"
    )
    ax.coastlines(color="grey")
    add_ticks(ax, x_ticks, y_ticks, left=False, bottom=draw_x_ticks)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax = axs[3]
    m_sp = ax.imshow(
        surface_precip, norm=precip_norm, cmap=precip_cmap, extent=ext, origin="upper"
    )
    ax.coastlines(color="grey")
    add_ticks(ax, x_ticks, y_ticks, left=False, bottom=draw_x_ticks)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    if axs[0] is not None:
        plt.colorbar(m_tbs, label="Brightness temperatures [K]", cax=axs[0])
        axs[0].yaxis.set_ticks_position("left")
        axs[0].yaxis.set_label_position("left")

    if axs[-1] is not None:
        plt.colorbar(m_sp, label="Surface precipitation [mm h$^{-1}$]", cax=axs[-1])

    return axs
