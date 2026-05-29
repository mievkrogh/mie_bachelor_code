# Import Packages
from __future__ import annotations
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pymap3d as pm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from IPython.display import display, HTML

import os
import sys
import imageio.v2 as imageio
import glob
import open3d as o3d
import healpix as hp

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from geopy.distance import great_circle
from scipy.interpolate import CubicSpline

from matplotlib.patches import Polygon
from astropy_healpix import HEALPix
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.cm import ScalarMappable

import gzip
import shutil
import os
from pathlib import Path

from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError, URLError
import tempfile
import plotly.graph_objects as go
from shapely.geometry import Point
from shapely.strtree import STRtree
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

import cartopy.feature as cfeature
from shapely.geometry import LineString, MultiLineString

from bokeh.plotting import figure, show, output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Slider,
    Select,
    CustomJS,
    Div,
    CheckboxGroup,
    HoverTool,
    Legend,
    LegendItem, WheelZoomTool, PanTool, ResetTool, SaveTool, Button
)
from bokeh.layouts import column

from collective0 import build_ipp_validation_df, ipp_pipeline_one_station, ipp_pipeline_multiple_stations, generate_ismr_paths_by_station
from ipp_on_map import plot_ipp_difference_magnitude, plot_ipp_map_geodetic, plot_ipp_sp3, plot_ipp_sp3_colored_by_time, make_ipp_sp3_gif, compute_geodetic_differences_ipp
from ToD_grid import define_laea_projection, build_laea_solution2_hierarchy
laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(projection="3574")

def read_station_data(csv_path_stations, lat_south, transformer_wgs84_to_laea):
    """
    Read station CSV, filter by latitude and add LAEA coordinates.
    """

    stations_df = pd.read_csv(csv_path_stations)

    if "latitude" not in stations_df.columns:
        raise KeyError("Station file must contain column: latitude")
    if "longitude" not in stations_df.columns:
        raise KeyError("Station file must contain column: longitude")

    stations_df = stations_df.copy()
    stations_df["latitude"] = pd.to_numeric(stations_df["latitude"], errors="coerce")
    stations_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")

    stations_df = stations_df.dropna(subset=["latitude", "longitude"])
    stations_df = stations_df[stations_df["latitude"] >= lat_south].copy()

    stations_df["x_laea"], stations_df["y_laea"] = transformer_wgs84_to_laea.transform(
        stations_df["longitude"].values,
        stations_df["latitude"].values
    )

    if "station" not in stations_df.columns:
        if "Station" in stations_df.columns:
            stations_df["station"] = stations_df["Station"]
        elif "name" in stations_df.columns:
            stations_df["station"] = stations_df["name"]
        else:
            stations_df["station"] = stations_df.index.astype(str)

    return stations_df

def collect_ismr_paths_from_extracted_folder(
    root_folder,
    extracted_folder_name="extracted_gz",
    file_extension=".ismr",
    station_id_length=4
):
    """
    Collect ISMR file paths from a shared extracted folder.

    Expected structure:
        root_folder/
            extracted_gz/
                SNOR019t00.26_.ismr
                THU3019t00.26_.ismr
                ...

    Returns
    -------
    paths_by_station : dict
        Dictionary where keys are station names and values are lists of file paths.

        Example:
        {
            "SNOR": [".../SNOR019t00.26_.ismr"],
            "THU3": [".../THU3019t00.26_.ismr"]
        }
    """

    root_folder = Path(root_folder)
    extracted_folder = root_folder / extracted_folder_name

    if not extracted_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {extracted_folder}")

    paths_by_station = {}

    for file_path in sorted(extracted_folder.glob(f"*{file_extension}")):
        station = file_path.name[:station_id_length]

        if station not in paths_by_station:
            paths_by_station[station] = []

        paths_by_station[station].append(str(file_path))

    return paths_by_station

def assign_ipp_to_one_grid_level(
    df_points,
    grid_df,
    transformer_wgs84_to_laea,
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    output_col="cell_id"
):
    """
    Assign IPP points to one LAEA grid level.
    """

    df = df_points.copy()

    df["ipp_x_laea"] = np.nan
    df["ipp_y_laea"] = np.nan
    df[output_col] = pd.NA

    valid_points = df[[lon_col, lat_col]].notna().all(axis=1)

    x, y = transformer_wgs84_to_laea.transform(
        df.loc[valid_points, lon_col].to_numpy(),
        df.loc[valid_points, lat_col].to_numpy()
    )

    df.loc[valid_points, "ipp_x_laea"] = x
    df.loc[valid_points, "ipp_y_laea"] = y

    # Remove padded NaN rows from your grid dataframe
    grid_valid = grid_df.dropna(subset=["cell_id", "polygon"]).reset_index(drop=True)

    polygons = list(grid_valid["polygon"])
    tree = STRtree(polygons)

    geom_to_index = {id(geom): i for i, geom in enumerate(polygons)}

    for idx in df.index[valid_points]:
        point = Point(df.loc[idx, "ipp_x_laea"], df.loc[idx, "ipp_y_laea"])

        candidates = tree.query(point)

        matched_indices = []

        for candidate in candidates:
            # Shapely 2.x returns indices
            if isinstance(candidate, (int, np.integer)):
                grid_idx = candidate
                polygon = polygons[grid_idx]

            # Shapely 1.x returns geometries
            else:
                polygon = candidate
                grid_idx = geom_to_index[id(candidate)]

            if polygon.covers(point):
                matched_indices.append(grid_idx)

        if len(matched_indices) > 0:
            # If point lies exactly on a boundary, choose one cell deterministically
            chosen_idx = sorted(
                matched_indices,
                key=lambda i: grid_valid.iloc[i]["cell_id"]
            )[0]

            df.loc[idx, output_col] = grid_valid.iloc[chosen_idx]["cell_id"]

    return df

def assign_ipp_to_solution2_hierarchy(
    df_points,
    cell_size,
    lat_south,
    transformer_wgs84_to_laea,
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat"
):
    """
    Assign IPP points to large, medium and small cells in the Solution 2 hierarchy.

    Changing cell_size changes the grid and therefore also the assigned cell IDs.
    """

    # Define Arctic clipping radius from lat_south
    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    # Build grid hierarchy
    grid_large, grid_medium, grid_small = build_laea_solution2_hierarchy(
        cell_size=cell_size,
        arctic_radius=arctic_radius
    )

    # Assign large cell IDs
    df_assigned = assign_ipp_to_one_grid_level(
        df_points=df_points,
        grid_df=grid_large,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="cell_id_large"
    )

    # Assign medium cell IDs
    df_assigned = assign_ipp_to_one_grid_level(
        df_points=df_assigned,
        grid_df=grid_medium,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="cell_id_medium"
    )

    # Assign small cell IDs
    df_assigned = assign_ipp_to_one_grid_level(
        df_points=df_assigned,
        grid_df=grid_small,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        lon_col=lon_col,
        lat_col=lat_col,
        output_col="cell_id_small"
    )

    return df_assigned, grid_large, grid_medium, grid_small

def plot_arctic_solution2_level_with_ipp(
    df_val_all,
    csv_path_stations,
    cell_size=1_200_000,
    determine_cell_size="medium",
    lat_south=50,
    lat_north=90,
    projection="3574",
    ipp_lon_col="ipp_sp3_lon",
    ipp_lat_col="ipp_sp3_lat",
    time_col=None,
    selected_time=None,
    station_col=None,
    selected_station=None,
    svid_col=None,
    selected_svids=None,
    ipp_color="orange",
    ipp_size=12,
    ipp_alpha=0.85,
    grid_cell_label=False,
    grid_cell_label_fontsize=10,
    show=True
):
    """
    Plot IPP points from df_val_all on top of the Arctic Solution 2 LAEA grid map.

    IPP coordinates are assumed to be geodetic lon/lat.
    """

    fig, ax, grid_df, stations_df = plot_arctic_map_with_laea_grid_solution2_level(
        csv_path_stations=csv_path_stations,
        cell_size=cell_size,
        determine_cell_size=determine_cell_size,
        lat_south=lat_south,
        lat_north=lat_north,
        projection=projection,
        grid_cell_label=grid_cell_label,
        grid_cell_label_fontsize=grid_cell_label_fontsize,
        show=False
    )

    ipp_df = df_val_all.copy()

    required_cols = [ipp_lon_col, ipp_lat_col]
    missing = [col for col in required_cols if col not in ipp_df.columns]
    if missing:
        raise ValueError(f"Missing required IPP columns: {missing}")

    ipp_df = ipp_df.dropna(subset=[ipp_lon_col, ipp_lat_col]).copy()

    # Optional time filter
    if time_col is not None and selected_time is not None:
        if time_col not in ipp_df.columns:
            raise ValueError(f"time_col='{time_col}' not found in df_val_all.")

        ipp_df[time_col] = pd.to_datetime(ipp_df[time_col])
        selected_time = pd.Timestamp(selected_time)

        ipp_df = ipp_df[ipp_df[time_col] == selected_time].copy()

    # Optional station filter
    if station_col is not None and selected_station is not None:
        if station_col not in ipp_df.columns:
            raise ValueError(f"station_col='{station_col}' not found in df_val_all.")

        ipp_df = ipp_df[
            ipp_df[station_col].astype(str) == str(selected_station)
        ].copy()

    # Optional SVID filter
    if svid_col is not None and selected_svids is not None:
        if svid_col not in ipp_df.columns:
            raise ValueError(f"svid_col='{svid_col}' not found in df_val_all.")

        if not isinstance(selected_svids, (list, tuple, set, np.ndarray, pd.Series)):
            selected_svids = [selected_svids]

        selected_svids = set(
            pd.Series(list(selected_svids))
            .dropna()
            .astype(str)
        )

        ipp_df = ipp_df[
            ipp_df[svid_col].astype(str).isin(selected_svids)
        ].copy()

    # Arctic extent filter
    ipp_df = ipp_df[
        (ipp_df[ipp_lat_col] >= lat_south) &
        (ipp_df[ipp_lat_col] <= lat_north)
    ].copy()

    ax.scatter(
        ipp_df[ipp_lon_col].values,
        ipp_df[ipp_lat_col].values,
        s=ipp_size,
        c=ipp_color,
        alpha=ipp_alpha,
        marker="o",
        edgecolors=ipp_color,
        linewidths=0.25,
        transform=ccrs.PlateCarree(),
        zorder=9,
        label="IPP"
    )

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    if selected_time is not None:
        ax.set_title(
            f"IPP points at {pd.Timestamp(selected_time).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            fontsize=13,
            pad=12
        )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_df, stations_df, ipp_df

# --------------------------------------------------
# Create CHAIN station CSV
# --------------------------------------------------
def create_chain_station_csv(out_root):
    """
    Create CHAIN station coordinate CSV from known CHAIN station metadata.
    Longitudes are converted from 0-360 °E to conventional -180 to 180 degrees.
    Station codes are converted to the 4-character codes used in CHAIN ISMR filenames.
    """

    data = [
        ("Arctic Bay", "ARCC", 73.004093, 274.973959),
        ("Arviat", "ARVC", 61.097941, 265.928533),
        ("Cambridge Bay", "CBBC", 69.101929, 254.884829),
        ("Churchill", "CHUC", 58.759279, 265.913402),
        ("Coral Harbour", "CORC", 64.188201, 276.650145),
        ("Dawson City", "DAWC", 64.049559, 220.888817),
        ("Eureka", "EURC", 79.990089, 274.097557),
        ("Fort McMurray", "MCMC", 56.649535, 248.779728),
        ("Fort Simpson", "FSIC", 61.756554, 238.771946),
        ("Fort Smith", "FSMC", 60.026095, 248.067109),
        ("Gillam", "GILC", 56.376600, 265.356197),
        ("Gjoa Haven", "GJOC", 68.632630, 264.151719),
        ("Grise Fiord", "GRIC", 76.423281, 277.096506),
        ("Hall Beach", "HALC", 68.767279, 278.743539),
        ("Iqaluit", "IQAC", 63.737377, 291.459735),
        ("Kugluktuk", "KUGC", 67.817781, 244.865276),
        ("Ministik Lake", "EDMC", 53.350818, 247.026160),
        ("Pond Inlet", "PONC", 72.693166, 282.044956),
        ("Qikiqtarjuaq", "QIKC", 67.559326, 295.966340),
        ("Rabbit Lake", "RABC", 58.226935, 256.322945),
        ("Rankin Inlet", "RANC", 62.824700, 267.885291),
        ("Repulse Bay", "REPC", 66.523589, 273.768972),
        ("Resolute", "RESC", 74.746627, 264.997469),
        ("Sachs Harbour", "SACC", 71.990630, 234.739381),
        ("Sanikiluaq", "SANC", 56.536360, 280.768771),
        ("Taloyoak", "TALC", 69.540923, 266.443335),
    ]

    df = pd.DataFrame(data, columns=["station_name", "station", "latitude", "lon_360"])

    df["longitude"] = ((df["lon_360"] + 180) % 360) - 180
    df["Height [m]"] = 0.0

    df = df[["station", "station_name", "latitude", "longitude", "Height [m]", "lon_360"]]

    chain_folder = Path(out_root) 
    chain_folder.mkdir(parents=True, exist_ok=True)

    out_path = chain_folder / "CHAIN_stations.csv"
    df.to_csv(out_path, index=False)

    return df, str(out_path)

def add_swado_chain_stations(
    ax,
    path_stations_swado,
    path_stations_chain,
    transformer_wgs84_to_laea,
    map_projection,
    lat_south=50,
    lon_col="Longitude",
    lat_col="Latitude",
    marker_size=55
):
    """
    Add SWADO and CHAIN stations with different colours and without labels.
    """

    df_swado = pd.read_csv(path_stations_swado)
    df_chain = pd.read_csv(path_stations_chain)

    df_swado["network"] = "SWADO"
    df_chain["network"] = "CHAIN"

    df_stations = pd.concat([df_swado, df_chain], ignore_index=True)

    df_stations = df_stations.dropna(subset=[lon_col, lat_col, "network"]).copy()
    df_stations = df_stations[df_stations[lat_col] >= lat_south].copy()

    x, y = transformer_wgs84_to_laea.transform(
        df_stations[lon_col].to_numpy(),
        df_stations[lat_col].to_numpy()
    )

    df_stations["x_laea"] = x
    df_stations["y_laea"] = y

    station_colours = {
        "SWADO": "purple",
        "CHAIN": "blue"
    }

    for network, colour in station_colours.items():
        df_net = df_stations[df_stations["network"] == network]

        ax.scatter(
            df_net["x_laea"],
            df_net["y_laea"],
            s=marker_size,
            marker="*",
            color=colour,
            edgecolor=None,
            linewidth=0.5,
            transform=map_projection,
            zorder=25,
            label=f"{network} stations"
        )

    return df_stations

def plot_ipp_count_per_cell_at_time_with_network_stations(
    df_assigned,
    path_stations_swado,
    path_stations_chain,
    cell_size,
    determine_cell_size="medium",
    time_to_plot=None,
    time_col="UTC Time",
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    station_lon_col="longitude",
    station_lat_col="latitude",
    lat_south=50,
    lat_north=90,
    projection="3574",
    show_cell_labels=False,
    show_count_labels=False,
    show_ipp_points=True,
    ipp_marker_size=18,
    ipp_color="grey",
    show=True
):
    """
    Plot number of IPPs per grid cell at one timestamp.

    The plot includes:
    - cells coloured by number of IPPs
    - IPP points at the selected timestamp
    - SWADO stations in red
    - CHAIN stations in blue
    - no station name labels
    - no white text boxes on map annotations

    Colour rule:
        red    : n < 3
        yellow : 3 <= n < 5
        green  : n >= 5
    """

    if determine_cell_size == "large":
        cell_col = "cell_id_large"
    elif determine_cell_size == "medium":
        cell_col = "cell_id_medium"
    elif determine_cell_size == "small":
        cell_col = "cell_id_small"
    else:
        raise ValueError("determine_cell_size must be 'large', 'medium', or 'small'")

    df = df_assigned.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if time_to_plot is None:
        time_to_plot = df[time_col].min()
    else:
        time_to_plot = pd.to_datetime(time_to_plot)

    # Use one station file only for building the base map.
    # We remove its labels afterwards and plot both station networks manually.
    fig, ax, grid_df_plot, stations_df_base = plot_arctic_map_with_laea_grid_solution2_level(
        csv_path_stations=path_stations_swado,
        cell_size=cell_size,
        determine_cell_size=determine_cell_size,
        lat_south=lat_south,
        lat_north=lat_north,
        projection=projection,
        grid_cell_label=False,
        show=False)

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    df_stations = add_swado_chain_stations(
        ax=ax,
        path_stations_swado=path_stations_swado,
        path_stations_chain=path_stations_chain,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        map_projection=map_projection,
        lat_south=lat_south,
        lon_col=station_lon_col,
        lat_col=station_lat_col,
        marker_size=55
    )

    # Select IPPs at selected time
    df_time = df[df[time_col] == time_to_plot].dropna(subset=[cell_col]).copy()

    count_df = (
        df_time.groupby(cell_col)
        .size()
        .reset_index(name="n_ipp")
    )

    # Merge counts into grid
    grid_plot = grid_df_plot.dropna(subset=["cell_id", "polygon"]).copy()

    grid_plot = grid_plot.merge(
        count_df,
        left_on="cell_id",
        right_on=cell_col,
        how="left"
    )

    grid_plot["n_ipp"] = grid_plot["n_ipp"].fillna(0).astype(int)

    # Same colour scale for map and colourbar
    from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
    from matplotlib.cm import ScalarMappable

    cell_alpha = 0.45

    colors = [
        to_rgba("red", alpha=cell_alpha),
        to_rgba("green", alpha=cell_alpha)
    ]

    cmap = ListedColormap(colors)

    vmax = max(6, int(grid_plot["n_ipp"].max()) + 1)
    bounds = [0, 4, vmax]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot cells
    # Cells with n_ipp == 0 are not coloured
    for _, row in grid_plot.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy

        if row["n_ipp"] == 0:
            facecolor = "none"
            alpha = 0.45
            zorder = 4
        else:
            facecolor = cmap(norm(row["n_ipp"]))
            alpha = 0.45
            zorder = 5

        ax.fill(
            x_cell,
            y_cell,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            alpha=alpha,
            transform=map_projection,
            zorder=zorder
        )

        if show_count_labels:
            ax.text(
                row["center_x"],
                row["center_y"],
                str(row["n_ipp"]),
                transform=map_projection,
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                zorder=10
            )

        if show_cell_labels:
            ax.text(
                row["center_x"],
                row["center_y"],
                str(row["cell_id"]),
                transform=map_projection,
                fontsize=6,
                ha="center",
                va="bottom",
                color="black",
                zorder=11
            )

    # Plot IPP points for selected time
    if show_ipp_points:
        df_ipp = df_time.dropna(subset=[lon_col, lat_col]).copy()

        ax.scatter(
            df_ipp[lon_col],
            df_ipp[lat_col],
            s=ipp_marker_size,
            color=ipp_color,
            edgecolor="grey",
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=30,
            label="IPP"
        )

    # Colourbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.04,
        ticks=[2, (4 + vmax) / 2]
    )

    cbar.set_label("Number of IPPs per cell", fontsize=18)
    cbar.ax.set_yticklabels([
        "1–3",
        f"≥4"
    ])

    # ax.set_title(
    #     f"IPP count per {determine_cell_size} grid cell\nUTC Time: {time_to_plot}",
    #     fontsize=16,
    #     fontweight="bold"
    # )

    ax.legend(loc="lower right", fontsize=15)

    # Crop top part of map
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + 0.70 * (ymax - ymin))

    # Remove white boxes behind latitude/longitude labels
    for txt in ax.texts:
        t = txt.get_text()

        if "°" in t:
            txt.set_bbox(None)

    # Remove latitude/longitude labels outside the visible y-limits
    new_ymin, new_ymax = ax.get_ylim()

    for txt in list(ax.texts):
        x_txt, y_txt = txt.get_position()
        transform = txt.get_transform()

        try:
            # Convert text position to display coordinates
            x_disp, y_disp = transform.transform((x_txt, y_txt))

            # Convert display coordinates back to data coordinates
            x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))

            if y_data < new_ymin or y_data > new_ymax:
                txt.remove()

        except Exception:
            pass

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_plot, count_df, df_time, df_stations

# Tilføj dTEC_dt_abs til df
def add_dTEC_dt_abs(
    df,
    dtec_col="dTEC",
    output_col="dTEC_dt_abs",
    interval_seconds=15
):
    """
    Compute |dTEC/dt| from ISMR dTEC values.

    ISMR definition:
        dTEC = TEC(t) - TEC(t-15s)

    Therefore:
        dTEC_dt = dTEC / 15

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ISMR data.

    dtec_col : str, default="dTEC"
        Column with dTEC values from ISMR.

    output_col : str, default="dTEC_dt_abs"
        Name of output column.

    interval_seconds : int, default=15
        Time interval used in ISMR (typically 15s).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added |dTEC/dt| column.
    """

    df = df.copy()

    # Ensure numeric
    df[dtec_col] = pd.to_numeric(df[dtec_col], errors="coerce")

    # Compute |dTEC/dt|
    df[output_col] = (df[dtec_col] / interval_seconds).abs()

    return df

# Compute simplified AIMS I_T per IPP observation
def compute_I_T_AIMS(
    df,
    col_S4="S4",
    col_phi="Phi60",
    col_TEC="TEC",
    col_dTEC_abs="dTEC_dt_abs",
    k=4.0,
    x0=1.0,
    S4_ref=0.3,
    phi_ref=0.2,
    TEC_ref=10.0,
    dTEC_ref=0.5,
    output_col="I_T"
):
    """
    Compute simplified AIMS I_T per IPP observation.

    I_T is based on normalized S4, Phi60, TEC and absolute dTEC/dt.
    Each normalized variable is passed through a logistic saturation function.
    """

    df = df.copy()

    required_cols = [col_S4, col_phi, col_TEC, col_dTEC_abs]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def sat(x):
        z = -k * (x - x0)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(z))

    x_S4 = df[col_S4] / S4_ref
    x_phi = df[col_phi] / phi_ref
    x_TEC = df[col_TEC] / TEC_ref
    x_dTEC = df[col_dTEC_abs] / dTEC_ref

    df[output_col] = (
        sat(x_S4) +
        sat(x_phi) +
        sat(x_TEC) +
        sat(x_dTEC)
    ) / 4

    return df

# Add AIMS I_T colour classification
def add_I_T_color_AIMS(
    df,
    I_T_col="I_T",
    color_col="I_T_color"
):
    """
    Add AIMS operational colour classification based on I_T.

    Green  : I_T < 0.35
    Yellow : 0.35 <= I_T < 0.65
    Red    : I_T >= 0.65
    """

    df = df.copy()

    if I_T_col not in df.columns:
        raise KeyError(f"Missing required column: {I_T_col}")

    df[I_T_col] = pd.to_numeric(df[I_T_col], errors="coerce")

    conditions = [
        df[I_T_col] < 0.35,
        (df[I_T_col] >= 0.35) & (df[I_T_col] < 0.65),
        df[I_T_col] >= 0.65
    ]

    colors = ["green", "yellow", "red"]

    df[color_col] = np.select(conditions, colors, default="lightgrey")

    return df

# Full pipeline to compute dTEC_dt_abs, I_T and AIMS colour class
def compute_full_I_T_pipeline_AIMS(df):
    """
    Compute dTEC_dt_abs, I_T and AIMS colour class.
    """

    df = add_dTEC_dt_abs(df)
    df = compute_I_T_AIMS(df)
    df = add_I_T_color_AIMS(df)

    return df

def plot_AIMS_IT_per_cell_at_time(
    df_assigned,
    path_stations_swado,
    path_stations_chain,
    cell_size,
    determine_cell_size="medium",
    time_to_plot=None,
    time_col="UTC Time",
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    station_lon_col="longitude",
    station_lat_col="latitude",
    lat_south=50,
    lat_north=90,
    projection="3574",
    it_col="I_T",
    col_S4="S4",
    col_phi="Phi60",
    col_TEC="TEC",
    col_dTEC_abs="dTEC_dt_abs",
    compute_it=True,
    min_ipp_per_cell=1,
    show_ipp_points=True,
    ipp_marker_size=16,
    ipp_color="grey",
    show=True
):
    """
    Compute and plot AIMS I_T per grid cell at one timestamp.

    AIMS classification:
        green  : I_T < 0.35
        yellow : 0.35 <= I_T < 0.65
        red    : I_T >= 0.65
    """

    if determine_cell_size == "large":
        cell_col = "cell_id_large"
    elif determine_cell_size == "medium":
        cell_col = "cell_id_medium"
    elif determine_cell_size == "small":
        cell_col = "cell_id_small"
    else:
        raise ValueError("determine_cell_size must be 'large', 'medium', or 'small'")

    df = df_assigned.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    if col_dTEC_abs not in df.columns:
        df = add_dTEC_dt_abs(
            df,
            dtec_col="dTEC",
            output_col=col_dTEC_abs,
            interval_seconds=15
        )

    if compute_it or it_col not in df.columns:
        df = compute_I_T_AIMS(
            df,
            col_S4=col_S4,
            col_phi=col_phi,
            col_TEC=col_TEC,
            col_dTEC_abs=col_dTEC_abs,
            output_col=it_col
        )

    if time_to_plot is None:
        time_to_plot = df[time_col].min()
    else:
        time_to_plot = pd.to_datetime(time_to_plot)

    fig, ax, grid_df_plot, stations_df_base = plot_arctic_map_with_laea_grid_solution2_level(
        csv_path_stations=path_stations_swado,
        cell_size=cell_size,
        determine_cell_size=determine_cell_size,
        lat_south=lat_south,
        lat_north=lat_north,
        projection=projection,
        grid_cell_label=False,
        show=False
    )

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    df_stations = add_swado_chain_stations(
        ax=ax,
        path_stations_swado=path_stations_swado,
        path_stations_chain=path_stations_chain,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        map_projection=map_projection,
        lat_south=lat_south,
        lon_col=station_lon_col,
        lat_col=station_lat_col,
        marker_size=55
    )

    df_time = df[df[time_col] == time_to_plot].dropna(subset=[cell_col, it_col]).copy()

    cell_stats = (
        df_time.groupby(cell_col)
        .agg(
            I_T_mean=(it_col, "mean"),
            I_T_median=(it_col, "median"),
            I_T_max=(it_col, "max"),
            n_ipp=(it_col, "count")
        )
        .reset_index()
    )

    cell_stats = cell_stats[cell_stats["n_ipp"] >= min_ipp_per_cell].copy()

    grid_plot = grid_df_plot.dropna(subset=["cell_id", "polygon"]).copy()

    grid_plot = grid_plot.merge(
        cell_stats,
        left_on="cell_id",
        right_on=cell_col,
        how="left"
    )

    # AIMS colour thresholds
    cell_alpha = 0.45

    cmap = ListedColormap([
        to_rgba("green", alpha=cell_alpha),
        to_rgba("yellow", alpha=cell_alpha),
        to_rgba("red", alpha=cell_alpha)
    ])

    bounds = [0.0, 0.35, 0.65, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)

    for _, row in grid_plot.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy

        if pd.isna(row["I_T_mean"]):
            facecolor = "none"
            zorder = 4
        else:
            facecolor = cmap(norm(row["I_T_mean"]))
            zorder = 5

        ax.fill(
            x_cell,
            y_cell,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            transform=map_projection,
            zorder=zorder
        )

    if show_ipp_points:
        df_ipp = df_time.dropna(subset=[lon_col, lat_col]).copy()

        ax.scatter(
            df_ipp[lon_col],
            df_ipp[lat_col],
            s=ipp_marker_size,
            color=ipp_color,
            edgecolor="grey",
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=30,
            label="IPP"
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.04,
        ticks=[0.175, 0.50, 0.825]
    )

    cbar.set_label("AIMS mean $I_T$ per cell", fontsize=12)
    cbar.ax.set_yticklabels([
        "$I_T < 0.35$",
        "$0.35 \\leq I_T < 0.65$",
        "$I_T \\geq 0.65$"
    ])

    ax.set_title(
        f"AIMS mean $I_T$ per {determine_cell_size} grid cell\nUTC Time: {time_to_plot}",
        fontsize=16,
        fontweight="bold"
    )

    ax.legend(loc="upper left", fontsize=13)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + 0.70 * (ymax - ymin))

    for txt in ax.texts:
        if "°" in txt.get_text():
            txt.set_bbox(None)

    new_ymin, new_ymax = ax.get_ylim()

    for txt in list(ax.texts):
        x_txt, y_txt = txt.get_position()
        transform = txt.get_transform()

        try:
            x_disp, y_disp = transform.transform((x_txt, y_txt))
            x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))

            if y_data < new_ymin or y_data > new_ymax:
                txt.remove()

        except Exception:
            pass

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_plot, cell_stats, df_time, df_stations, df

def plot_AIMS_IT_per_cell_at_time_with_colored_IPPs(
    df_assigned,
    path_stations_swado,
    path_stations_chain,
    cell_size,
    determine_cell_size="medium",
    time_to_plot=None,
    time_col="UTC Time",
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    station_lon_col="longitude",
    station_lat_col="latitude",
    lat_south=50,
    lat_north=90,
    projection="3574",
    it_col="I_T",
    col_S4="S4",
    col_phi="Phi60",
    col_TEC="TEC",
    col_dTEC_abs="dTEC_dt_abs",
    compute_it=True,
    min_ipp_per_cell=1,
    show_ipp_points=True,
    ipp_marker_size=20,
    show=True
):
    """
    Plot AIMS I_T per grid cell at one timestamp.

    Cell colour:
        based on mean I_T per cell.

    IPP colour:
        based on individual IPP-level I_T_color.
    """

    if determine_cell_size == "large":
        cell_col = "cell_id_large"
    elif determine_cell_size == "medium":
        cell_col = "cell_id_medium"
    elif determine_cell_size == "small":
        cell_col = "cell_id_small"
    else:
        raise ValueError("determine_cell_size must be 'large', 'medium', or 'small'")

    df = df_assigned.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if col_dTEC_abs not in df.columns:
        df = add_dTEC_dt_abs(
            df,
            dtec_col="dTEC",
            output_col=col_dTEC_abs,
            interval_seconds=15
        )

    if compute_it or it_col not in df.columns:
        df = compute_I_T_AIMS(
            df,
            col_S4=col_S4,
            col_phi=col_phi,
            col_TEC=col_TEC,
            col_dTEC_abs=col_dTEC_abs,
            output_col=it_col
        )

    if "I_T_color" not in df.columns:
        df = add_I_T_color_AIMS(
            df,
            I_T_col=it_col,
            color_col="I_T_color",
            class_col="I_T_class"
        )

    if time_to_plot is None:
        time_to_plot = df[time_col].min()
    else:
        time_to_plot = pd.to_datetime(time_to_plot)

    fig, ax, grid_df_plot, stations_df_base = plot_arctic_map_with_laea_grid_solution2_level(
        csv_path_stations=path_stations_swado,
        cell_size=cell_size,
        determine_cell_size=determine_cell_size,
        lat_south=lat_south,
        lat_north=lat_north,
        projection=projection,
        grid_cell_label=False,
        show=False
    )

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    df_stations = add_swado_chain_stations(
        ax=ax,
        path_stations_swado=path_stations_swado,
        path_stations_chain=path_stations_chain,
        transformer_wgs84_to_laea=transformer_wgs84_to_laea,
        map_projection=map_projection,
        lat_south=lat_south,
        lon_col=station_lon_col,
        lat_col=station_lat_col,
        marker_size=55
    )

    df_time = df[df[time_col] == time_to_plot].dropna(subset=[cell_col, it_col]).copy()

    cell_stats = (
        df_time.groupby(cell_col)
        .agg(
            I_T_mean=(it_col, "mean"),
            I_T_median=(it_col, "median"),
            I_T_max=(it_col, "max"),
            n_ipp=(it_col, "count")
        )
        .reset_index()
    )

    cell_stats = cell_stats[cell_stats["n_ipp"] >= min_ipp_per_cell].copy()

    grid_plot = grid_df_plot.dropna(subset=["cell_id", "polygon"]).copy()

    grid_plot = grid_plot.merge(
        cell_stats,
        left_on="cell_id",
        right_on=cell_col,
        how="left"
    )

    cell_alpha = 0.45

    cmap = ListedColormap([
        to_rgba("green", alpha=cell_alpha),
        to_rgba("yellow", alpha=cell_alpha),
        to_rgba("red", alpha=cell_alpha)
    ])

    bounds = [0.0, 0.35, 0.65, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)

    for _, row in grid_plot.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy

        if pd.isna(row["I_T_mean"]):
            facecolor = "none"
            zorder = 4
        else:
            facecolor = cmap(norm(row["I_T_mean"]))
            zorder = 5

        ax.fill(
            x_cell,
            y_cell,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            transform=map_projection,
            zorder=zorder
        )

    if show_ipp_points:
        df_ipp = df_time.dropna(subset=[lon_col, lat_col, "I_T_color"]).copy()

        ax.scatter(
            df_ipp[lon_col],
            df_ipp[lat_col],
            s=ipp_marker_size,
            c=df_ipp["I_T_color"],
            edgecolor="black",
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            zorder=30,
            label="IPP-level $I_T$"
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.04,
        ticks=[0.175, 0.50, 0.825]
    )

    cbar.set_label("AIMS mean $I_T$ per cell", fontsize=12)
    cbar.ax.set_yticklabels([
        "$I_T < 0.35$",
        "$0.35 \\leq I_T < 0.65$",
        "$I_T \\geq 0.65$"
    ])

    ax.set_title(
        f"AIMS mean $I_T$ per {determine_cell_size} grid cell\nUTC Time: {time_to_plot}",
        fontsize=16,
        fontweight="bold"
    )

    ax.legend(loc="upper left", fontsize=13)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + 0.70 * (ymax - ymin))

    for txt in ax.texts:
        if "°" in txt.get_text():
            txt.set_bbox(None)

    new_ymin, new_ymax = ax.get_ylim()

    for txt in list(ax.texts):
        x_txt, y_txt = txt.get_position()
        transform = txt.get_transform()

        try:
            x_disp, y_disp = transform.transform((x_txt, y_txt))
            x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))

            if y_data < new_ymin or y_data > new_ymax:
                txt.remove()

        except Exception:
            pass

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_plot, cell_stats, df_time, df_stations, df

# --------------------------------------------------
# Create BOKEH plot 
# --------------------------------------------------
# Add Borderlines
def add_cartopy_feature_to_bokeh_laea(
    p,
    feature,
    transformer_wgs84_to_laea,
    line_color="gray",
    line_width=1,
    line_alpha=0.7
):
    """
    Add Cartopy coastline/border geometries to a Bokeh LAEA x/y plot.
    """

    xs_all = []
    ys_all = []

    for geom in feature.geometries():
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = list(geom.geoms)
        else:
            continue

        for line in geoms:
            lon, lat = line.xy

            x, y = transformer_wgs84_to_laea.transform(
                np.array(lon),
                np.array(lat)
            )

            xs_all.append(list(x))
            ys_all.append(list(y))

    source = ColumnDataSource(data=dict(xs=xs_all, ys=ys_all))

    renderer = p.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        line_color=line_color,
        line_width=line_width,
        line_alpha=line_alpha
    )

    return renderer

# classify I_T
def classify_IT_value(value):
    """
    Classify AIMS I_T value.
    """

    if pd.isna(value):
        return "No data"
    elif value < 0.35:
        return "Green"
    elif value < 0.65:
        return "Yellow"
    else:
        return "Red"

# Add Color 
def IT_class_to_color(class_name):
    """
    Convert AIMS I_T class to plot colour.
    """

    color_map = {
        "Green": "green",
        "Yellow": "yellow",
        "Red": "red",
        "No data": "white", 
        "Low coverage": "white"
    }

    return color_map.get(class_name, "white")

# Bokeh plot, with timeline and multilevel
def plot_laea_solution2_IT_timeline_multilevel_bokeh(
    df_ipp,
    cell_size,
    lat_south=50,
    projection="3574",
    initial_grid_level="medium",
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    time_col="UTC Time",
    it_value_col="I_T",
    time_freq="min",
    time_start=None,
    time_end=None,
    min_ipp_per_cell=4,
    width=950,
    height=950,
    grid_line_color="black",
    grid_line_width=0.8,
    grid_line_alpha=0.8,
    cell_fill_alpha=0.45,
    ipp_size=7,
    ipp_alpha=0.9,
    show_underlay=True,
    coastline_scale="50m",
    csv_path_swado=None,
    csv_path_chain=None,
    station_lon_col="longitude",
    station_lat_col="latitude",
    station_name_col="station",
    show_station_labels=False,
    show_north_pole=True,
):
    """
    Bokeh LAEA timeline plot with selectable grid level:
    - large
    - medium
    - small

    Cell colour:
        mean I_T_value per grid cell per frame_time.

    IPP colour:
        individual I_T_value.
    """

    if initial_grid_level not in ["large", "medium", "small"]:
        raise ValueError("initial_grid_level must be 'large', 'medium', or 'small'")

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    grid_large, grid_medium, grid_small = build_laea_solution2_hierarchy(
        cell_size=cell_size,
        arctic_radius=arctic_radius
    )

    grids = {
        "large": grid_large.dropna(subset=["polygon"]).copy(),
        "medium": grid_medium.dropna(subset=["polygon"]).copy(),
        "small": grid_small.dropna(subset=["polygon"]).copy()
    }

    cell_cols = {
        "large": "cell_id_large",
        "medium": "cell_id_medium",
        "small": "cell_id_small"
    }

    cell_size_km = {
        "large": cell_size / 1000,
        "medium": cell_size / 2 / 1000,
        "small": cell_size / 4 / 1000
    }

    df = df_ipp.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if time_start is not None:
        df = df[df[time_col] >= pd.to_datetime(time_start)].copy()

    if time_end is not None:
        df = df[df[time_col] <= pd.to_datetime(time_end)].copy()

    required_cols = [lon_col, lat_col, time_col, it_value_col, "Elevation (degrees)"]
    for level in ["large", "medium", "small"]:
        required_cols.append(cell_cols[level])

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df_ipp: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    if time_freq == "min":
        df["frame_time"] = df[time_col].dt.floor("min")
    elif time_freq == "5min":
        df["frame_time"] = df[time_col].dt.floor("5min")
    elif time_freq == "10min":
        df["frame_time"] = df[time_col].dt.floor("10min")
    elif time_freq is None:
        df["frame_time"] = df[time_col]
    else:
        raise ValueError("time_freq must be 'min', '5min', '10min', or None")

    frame_times = sorted(df["frame_time"].dropna().unique())

    if len(frame_times) == 0:
        raise ValueError("No valid frame times found.")

    ipp_x, ipp_y = transformer_wgs84_to_laea.transform(
        df[lon_col].values,
        df[lat_col].values
    )

    df["ipp_x_laea"] = ipp_x
    df["ipp_y_laea"] = ipp_y
    df["I_T_class"] = df[it_value_col].apply(classify_IT_value)
    df["I_T_color"] = df["I_T_class"].apply(IT_class_to_color)

    time_labels = [
        pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S")
        for t in frame_times
    ]

    # --------------------------------------------------
    # Precompute frames for all levels
    # --------------------------------------------------
    grid_frames_by_level = {}
    ipp_frames_by_level = {}

    for level in ["large", "medium", "small"]:
        grid_df = grids[level]
        cell_col = cell_cols[level]

        xs, ys, grid_cell_ids = [], [], []

        for _, row_i in grid_df.iterrows():
            x_poly, y_poly = row_i["polygon"].exterior.xy
            xs.append(list(x_poly))
            ys.append(list(y_poly))
            grid_cell_ids.append(str(row_i["cell_id"]))

        base_grid = pd.DataFrame({
            "cell_id": grid_cell_ids,
            "xs": xs,
            "ys": ys
        })

        grid_frames = {}
        ipp_frames = {}

        for i, t in enumerate(frame_times):
            df_t = df[df["frame_time"] == t].copy()

            cell_stats = (
                df_t
                .groupby(cell_col)
                .agg(
                    I_T_mean=(it_value_col, "mean"),
                    I_T_median=(it_value_col, "median"),
                    I_T_std=(it_value_col, "std"),
                    I_T_max=(it_value_col, "max"),
                    I_T_min=(it_value_col, "min"),
                    I_T_q25=(it_value_col, lambda x: x.quantile(0.25)),
                    I_T_q75=(it_value_col, lambda x: x.quantile(0.75)),
                    n_ipp=(it_value_col, "count")
                )
                .reset_index()
            )

            cell_stats["I_T_iqr"] = cell_stats["I_T_q75"] - cell_stats["I_T_q25"]
            cell_stats["I_T_range"] = cell_stats["I_T_max"] - cell_stats["I_T_min"]

            cell_stats["I_T_class"] = cell_stats["I_T_mean"].apply(classify_IT_value)
            cell_stats["I_T_color"] = cell_stats["I_T_class"].apply(IT_class_to_color)

            low_coverage = cell_stats["n_ipp"] < min_ipp_per_cell

            cell_stats.loc[low_coverage, "I_T_class"] = "Low coverage"
            cell_stats.loc[low_coverage, "I_T_color"] = "white"

            grid_t = base_grid.merge(
                cell_stats,
                left_on="cell_id",
                right_on=cell_col,
                how="left"
            )

            grid_t["n_ipp"] = grid_t["n_ipp"].fillna(0).astype(int)
            grid_t["I_T_class"] = grid_t["I_T_class"].fillna("No data")
            grid_t["I_T_color"] = grid_t["I_T_color"].fillna("white")
            
            for c in ["I_T_mean", "I_T_median", "I_T_std", "I_T_max", "I_T_min","I_T_q25", "I_T_q75", "I_T_iqr", "I_T_range"]:
                grid_t[c] = grid_t[c].astype(float)

            grid_frames[str(i)] = dict(
                xs=grid_t["xs"].tolist(),
                ys=grid_t["ys"].tolist(),
                cell_id=grid_t["cell_id"].astype(str).tolist(),
                n_ipp=grid_t["n_ipp"].tolist(),
                I_T_mean=grid_t["I_T_mean"].tolist(),
                I_T_median=grid_t["I_T_median"].tolist(),
                I_T_max=grid_t["I_T_max"].tolist(),
                I_T_min=grid_t["I_T_min"].tolist(),
                I_T_class=grid_t["I_T_class"].astype(str).tolist(),
                I_T_color=grid_t["I_T_color"].astype(str).tolist(), 
                I_T_std=grid_t["I_T_std"].tolist(),
                I_T_q25=grid_t["I_T_q25"].tolist(),
                I_T_q75=grid_t["I_T_q75"].tolist(),
                I_T_iqr=grid_t["I_T_iqr"].tolist(),
                I_T_range=grid_t["I_T_range"].tolist()
            )

            ipp_data = {
                "x": df_t["ipp_x_laea"].tolist(),
                "y": df_t["ipp_y_laea"].tolist(),
                "lon": df_t[lon_col].tolist(),
                "lat": df_t[lat_col].tolist(),
                "time": df_t[time_col].astype(str).tolist(),
                it_value_col: df_t[it_value_col].tolist(),
                "I_T_class": df_t["I_T_class"].astype(str).tolist(),
                "I_T_color": df_t["I_T_color"].astype(str).tolist(),
                "cell_id": df_t[cell_col].astype(str).tolist(), 
                "Elevation (degrees)": df_t["Elevation (degrees)"].tolist()
            }

            if "SVID" in df_t.columns:
                ipp_data["SVID"] = df_t["SVID"].astype(str).tolist()
            else:
                ipp_data["SVID"] = [""] * len(df_t)

            ipp_frames[str(i)] = ipp_data

        grid_frames_by_level[level] = grid_frames
        ipp_frames_by_level[level] = ipp_frames

    initial_index = "0"

    grid_source = ColumnDataSource(
        data=grid_frames_by_level[initial_grid_level][initial_index]
    )

    ipp_source = ColumnDataSource(
        data=ipp_frames_by_level[initial_grid_level][initial_index]
    )

    base_title = (
        f"Solution 2 grid with AIMS I_T timeline "
        f"| Level: {initial_grid_level} "
        f"| Cell size: {cell_size_km[initial_grid_level]:.0f} km"
    )

    p = figure(
        width=width,
        height=height,
        title=f"{base_title} | Time: {time_labels[0]}",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

        # --------------------------------------------------
    # Add SWADO / CHAIN stations and North Pole
    # --------------------------------------------------


    def add_stations_to_bokeh(p, csv_path, label, marker_color):
        if csv_path is None:
            return None, None

        stations = pd.read_csv(csv_path)

        x_sta, y_sta = transformer_wgs84_to_laea.transform(
            stations[station_lon_col].values,
            stations[station_lat_col].values
        )

        stations["x_laea"] = x_sta
        stations["y_laea"] = y_sta
        stations["network"] = label

        if station_name_col not in stations.columns:
            stations[station_name_col] = label

        source_sta = ColumnDataSource(stations)

        renderer = p.scatter(
            x="x_laea",
            y="y_laea",
            source=source_sta,
            marker="star",
            size=15,
            color=marker_color,
            line_color=marker_color,
            line_width=1.0,
            legend_label=label
        )

        hover_sta = HoverTool(
            renderers=[renderer],
            tooltips=[
                ("network", "@network"),
                ("station", f"@{{{station_name_col}}}"),
                ("lon", f"@{{{station_lon_col}}}{{0.00}}"),
                ("lat", f"@{{{station_lat_col}}}{{0.00}}")
            ]
        )

        return renderer, hover_sta


    swado_renderer, swado_hover = add_stations_to_bokeh(
        p=p,
        csv_path=csv_path_swado,
        label="SWADO",
        marker_color="purple"
    )

    chain_renderer, chain_hover = add_stations_to_bokeh(
        p=p,
        csv_path=csv_path_chain,
        label="CHAIN",
        marker_color="blue"
    )

    if show_north_pole:
        x_np, y_np = transformer_wgs84_to_laea.transform(0, 90)

        north_pole_source = ColumnDataSource(data=dict(
            x=[x_np],
            y=[y_np],
            name=["Geographic North Pole"]
        ))

        north_pole_renderer = p.scatter(
            x="x",
            y="y",
            source=north_pole_source,
            marker="asterisk",
            size=18,
            color="black",
            legend_label="North Pole"
        )

        north_pole_hover = HoverTool(
            renderers=[north_pole_renderer],
            tooltips=[
                ("point", "@name"),
                ("lat", "90.00"),
                ("lon", "0.00")
            ]
        )

        p.add_tools(north_pole_hover)

    p.xaxis.axis_label = "LAEA x [m]"
    p.yaxis.axis_label = "LAEA y [m]"

    if show_underlay:
        add_cartopy_feature_to_bokeh_laea(
            p,
            cfeature.COASTLINE.with_scale(coastline_scale),
            transformer_wgs84_to_laea,
            line_color="dimgray",
            line_width=1,
            line_alpha=0.8
        )

        add_cartopy_feature_to_bokeh_laea(
            p,
            cfeature.BORDERS.with_scale(coastline_scale),
            transformer_wgs84_to_laea,
            line_color="gray",
            line_width=0.7,
            line_alpha=0.6
        )

    cell_renderer = p.patches(
        xs="xs",
        ys="ys",
        source=grid_source,
        fill_color="I_T_color",
        fill_alpha=cell_fill_alpha,
        line_color=grid_line_color,
        line_width=grid_line_width,
        line_alpha=grid_line_alpha
    )

    ipp_renderer = p.scatter(
        x="x",
        y="y",
        source=ipp_source,
        size=ipp_size,
        color="I_T_color",
        alpha=ipp_alpha,
        line_color="black",
        line_width=0.5
    )

    cell_hover = HoverTool(
        renderers=[cell_renderer],
        mode="mouse",
        tooltips=[
            ("cell_id", "@cell_id"),
            ("n_ipp", "@n_ipp"),
            ("mean I_T", "@I_T_mean{0.000}"),
            ("median I_T", "@I_T_median{0.000}"),
            ("std I_T", "@I_T_std{0.000}"),
            ("Q25 I_T", "@I_T_q25{0.000}"),
            ("Q75 I_T", "@I_T_q75{0.000}"),
            ("IQR I_T", "@I_T_iqr{0.000}"),
            ("range I_T", "@I_T_range{0.000}"),
            ("max I_T", "@I_T_max{0.000}"),
            ("min I_T", "@I_T_min{0.000}"),
            ("class", "@I_T_class")
        ]
    )

    ipp_hover = HoverTool(
        renderers=[ipp_renderer],
        mode="mouse",
        tooltips=[
            ("time", "@time"),
            ("SVID", "@SVID"),
            ("Elevation (degrees)", "@{Elevation (degrees)}{0.0}"),
            ("lon", "@lon{0.00}"),
            ("lat", "@lat{0.00}"),
            (it_value_col, f"@{{{it_value_col}}}{{0.000}}"),
            ("class", "@I_T_class"),
            ("cell_id", "@cell_id")
        ]
    )

    tools_to_add = [cell_hover, ipp_hover]

    if swado_hover is not None:
        tools_to_add.append(swado_hover)

    if chain_hover is not None:
        tools_to_add.append(chain_hover)

    p.add_tools(*tools_to_add)

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    slider = Slider(
        start=0,
        end=len(frame_times) - 1,
        value=0,
        step=1,
        title=f"Time: {time_labels[0]}"
    )

    play_button = Button(label="Play", button_type="success", width=80)

    play_callback = CustomJS(
        args=dict(slider=slider, button=play_button),
        code="""
            if (button.label === "Play") {
                button.label = "Pause";

                button._interval = setInterval(function() {
                    if (slider.value < slider.end) {
                        slider.value = slider.value + 1;
                    } else {
                        slider.value = slider.start;
                    }
                }, 500);

            } else {
                button.label = "Play";
                clearInterval(button._interval);
            }
        """
    )

    play_button.js_on_click(play_callback)

    

    grid_select = Select(
        title="Grid level",
        value=initial_grid_level,
        options=["large", "medium", "small"]
    )
    callback = CustomJS(
        args=dict(
            grid_source=grid_source,
            ipp_source=ipp_source,
            grid_frames_by_level=grid_frames_by_level,
            ipp_frames_by_level=ipp_frames_by_level,
            time_labels=time_labels,
            plot=p,
            slider=slider,
            grid_select=grid_select,
            cell_size_km=cell_size_km, min_ipp_per_cell=min_ipp_per_cell
        ),
        code="""
            const level = grid_select.value;
            const i = slider.value.toString();

            grid_source.data = grid_frames_by_level[level][i];
            ipp_source.data = ipp_frames_by_level[level][i];

            grid_source.change.emit();
            ipp_source.change.emit();

            slider.title = "Time: " + time_labels[slider.value];

            const title =
                "Grid with I_T timeline" +
                " | Cell size: " + cell_size_km[level].toFixed(0) + " km" +
                " | Time: " + time_labels[slider.value] + 
                " | min IPP/cell: " + min_ipp_per_cell;

            plot.title.text = title;
        """
    )

    slider.js_on_change("value", callback)
    grid_select.js_on_change("value", callback)

    layout = column(row(grid_select, play_button, slider), p)

    return layout, p, slider, grid_select, grid_frames_by_level, ipp_frames_by_level, df

def compute_coverage_quality_by_level(
    df_all_assigned,
    time_col="UTC Time",
    time_freq="min",
    station_col="Station",
    svid_col="SVID",
    min_ipp_per_cell=4,
    n_good_large=12,
    n_good_medium=8,
    n_good_small=4,
    receiver_good=3,
    use_satellite=False,
    sat_good=4,
    weights=None
):
    """
    Compute coverage quality per cell, per time step, and per grid level.

    Grid levels:
        large, medium, small

    Coverage components:
        C_count:
            Number of IPPs in the cell.
            0 at min_ipp_per_cell, 1 at n_good or above.

        C_spread:
            Spatial spread inside the cell.
            For large cells: based on medium and small subcells.
            For medium cells: based on small subcells.
            For small cells: set to 1 if enough IPPs exist.

        C_receiver:
            Number of unique receiver stations.

        C_satellite:
            Optional. Number of unique satellites.

    Returns
    -------
    coverage_df : pandas.DataFrame
    """

    df = df_all_assigned.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # --------------------------------------------------
    # Time aggregation
    # --------------------------------------------------
    if time_freq == "min":
        df["frame_time"] = df[time_col].dt.floor("min")
    elif time_freq == "5min":
        df["frame_time"] = df[time_col].dt.floor("5min")
    elif time_freq == "10min":
        df["frame_time"] = df[time_col].dt.floor("10min")
    elif time_freq is None:
        df["frame_time"] = df[time_col]
    else:
        raise ValueError("time_freq must be 'min', '5min', '10min', or None")

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------
    required_cols = [
        "frame_time",
        "cell_id_large",
        "cell_id_medium",
        "cell_id_small",
        station_col
    ]

    if use_satellite:
        required_cols.append(svid_col)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    # --------------------------------------------------
    # Default weights
    # --------------------------------------------------
    if weights is None:
        if use_satellite:
            weights = {
                "count": 0.30,
                "spread": 0.50,
                "receiver": 0.10,
                "satellite": 0.10
            }
        else:
            weights = {
                "count": 0.35,
                "spread": 0.45,
                "receiver": 0.20
            }

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------
    def normalise_count(n_ipp, n_good):
        """
        0 at min_ipp_per_cell.
        1 at n_good or above.
        """
        if n_ipp < min_ipp_per_cell:
            return np.nan

        if n_good <= min_ipp_per_cell:
            return 1.0

        return min((n_ipp - min_ipp_per_cell) / (n_good - min_ipp_per_cell), 1.0)

    def normalise_receiver(n_receiver):
        """
        0 for one receiver.
        1 for receiver_good or above.
        """
        if n_receiver <= 1:
            return 0.0

        if receiver_good <= 1:
            return 1.0

        return min((n_receiver - 1) / (receiver_good - 1), 1.0)

    def normalise_satellite(n_sat):
        """
        0 for one satellite.
        1 for sat_good or above.
        """
        if n_sat <= 1:
            return 0.0

        if sat_good <= 1:
            return 1.0

        return min((n_sat - 1) / (sat_good - 1), 1.0)

    def entropy_uniformity(counts, n_possible_subcells):
        """
        Compute how evenly IPPs are distributed between subcells.

        Returns a value between 0 and 1:
            1 = perfectly even distribution
            0 = all observations concentrated in one subcell

        Empty subcells are included with count 0.
        """

        counts = np.asarray(counts, dtype=float)

        if len(counts) == 0:
            return np.nan

        if n_possible_subcells <= 1:
            return 1.0

        # Add zero-count subcells so the entropy is normalized against
        # the full number of possible subcells.
        if len(counts) < n_possible_subcells:
            counts = np.concatenate([
                counts,
                np.zeros(n_possible_subcells - len(counts))
            ])

        total = counts.sum()

        if total <= 0:
            return np.nan

        p = counts / total
        p = p[p > 0]

        entropy = -np.sum(p * np.log(p))
        entropy_max = np.log(n_possible_subcells)

        if entropy_max == 0:
            return 1.0

        return entropy / entropy_max

    def classify_coverage_quality(value):
        if pd.isna(value):
            return "No data"
        elif value < 0.30:
            return "Poor"
        elif value < 0.60:
            return "Moderate"
        else:
            return "Good"

    all_results = []

    # ==================================================
    # LARGE LEVEL
    # ==================================================
    large_stats = (
        df.groupby(["frame_time", "cell_id_large"])
        .agg(
            n_ipp=("cell_id_large", "count"),
            n_medium_subcells=("cell_id_medium", "nunique"),
            n_small_subcells=("cell_id_small", "nunique"),
            n_receivers=(station_col, "nunique")
        )
        .reset_index()
    )

    # Counts per medium subcell inside each large cell
    large_medium_counts = (
        df.groupby(["frame_time", "cell_id_large", "cell_id_medium"])
        .size()
        .reset_index(name="n_ipp_subcell")
    )

    large_uniform_medium = (
        large_medium_counts
        .groupby(["frame_time", "cell_id_large"])["n_ipp_subcell"]
        .apply(lambda counts: entropy_uniformity(counts, n_possible_subcells=4))
        .reset_index(name="C_uniform_medium")
    )

    # Counts per small subcell inside each large cell
    large_small_counts = (
        df.groupby(["frame_time", "cell_id_large", "cell_id_small"])
        .size()
        .reset_index(name="n_ipp_subcell")
    )

    large_uniform_small = (
        large_small_counts
        .groupby(["frame_time", "cell_id_large"])["n_ipp_subcell"]
        .apply(lambda counts: entropy_uniformity(counts, n_possible_subcells=16))
        .reset_index(name="C_uniform_small")
    )

    large_stats = large_stats.merge(
        large_uniform_medium,
        on=["frame_time", "cell_id_large"],
        how="left"
    )

    large_stats = large_stats.merge(
        large_uniform_small,
        on=["frame_time", "cell_id_large"],
        how="left"
    )

    if use_satellite:
        large_sat = (
            df.groupby(["frame_time", "cell_id_large"])[svid_col]
            .nunique()
            .reset_index(name="n_satellites")
        )

        large_stats = large_stats.merge(
            large_sat,
            on=["frame_time", "cell_id_large"],
            how="left"
        )

    large_stats["grid_level"] = "large"
    large_stats["cell_id"] = large_stats["cell_id_large"].astype(str)

    large_stats["C_count"] = large_stats["n_ipp"].apply(
        lambda n: normalise_count(n, n_good_large)
    )

    large_stats["C_coverage_medium"] = (large_stats["n_medium_subcells"] / 4).clip(0, 1)
    large_stats["C_coverage_small"] = (large_stats["n_small_subcells"] / 16).clip(0, 1)

    large_stats["C_spread"] = (
        0.25 * large_stats["C_coverage_medium"] +
        0.25 * large_stats["C_coverage_small"] +
        0.25 * large_stats["C_uniform_medium"] +
        0.25 * large_stats["C_uniform_small"]
    ).clip(0, 1)

    large_stats["C_receiver"] = large_stats["n_receivers"].apply(normalise_receiver)

    if use_satellite:
        large_stats["C_satellite"] = large_stats["n_satellites"].apply(normalise_satellite)

        large_stats["coverage_quality"] = (
            weights["count"] * large_stats["C_count"] +
            weights["spread"] * large_stats["C_spread"] +
            weights["receiver"] * large_stats["C_receiver"] +
            weights["satellite"] * large_stats["C_satellite"]
        )
    else:
        large_stats["coverage_quality"] = (
            weights["count"] * large_stats["C_count"] +
            weights["spread"] * large_stats["C_spread"] +
            weights["receiver"] * large_stats["C_receiver"]
        )

    all_results.append(large_stats)

    # ==================================================
    # MEDIUM LEVEL
    # ==================================================
    medium_stats = (
        df.groupby(["frame_time", "cell_id_medium"])
        .agg(
            n_ipp=("cell_id_medium", "count"),
            n_small_subcells=("cell_id_small", "nunique"),
            n_receivers=(station_col, "nunique")
        )
        .reset_index()
    )

    # Counts per small subcell inside each medium cell
    medium_small_counts = (
        df.groupby(["frame_time", "cell_id_medium", "cell_id_small"])
        .size()
        .reset_index(name="n_ipp_subcell")
    )

    medium_uniform_small = (
        medium_small_counts
        .groupby(["frame_time", "cell_id_medium"])["n_ipp_subcell"]
        .apply(lambda counts: entropy_uniformity(counts, n_possible_subcells=4))
        .reset_index(name="C_uniform_small")
    )

    medium_stats = medium_stats.merge(
        medium_uniform_small,
        on=["frame_time", "cell_id_medium"],
        how="left"
    )

    if use_satellite:
        medium_sat = (
            df.groupby(["frame_time", "cell_id_medium"])[svid_col]
            .nunique()
            .reset_index(name="n_satellites")
        )

        medium_stats = medium_stats.merge(
            medium_sat,
            on=["frame_time", "cell_id_medium"],
            how="left"
        )

    medium_stats["grid_level"] = "medium"
    medium_stats["cell_id"] = medium_stats["cell_id_medium"].astype(str)

    medium_stats["C_count"] = medium_stats["n_ipp"].apply(
        lambda n: normalise_count(n, n_good_medium)
    )

    medium_stats["C_coverage_small"] = (medium_stats["n_small_subcells"] / 4).clip(0, 1)

    medium_stats["C_spread"] = (
        0.50 * medium_stats["C_coverage_small"] +
        0.50 * medium_stats["C_uniform_small"]
    ).clip(0, 1)

    medium_stats["C_receiver"] = medium_stats["n_receivers"].apply(normalise_receiver)

    if use_satellite:
        medium_stats["C_satellite"] = medium_stats["n_satellites"].apply(normalise_satellite)

        medium_stats["coverage_quality"] = (
            weights["count"] * medium_stats["C_count"] +
            weights["spread"] * medium_stats["C_spread"] +
            weights["receiver"] * medium_stats["C_receiver"] +
            weights["satellite"] * medium_stats["C_satellite"]
        )
    else:
        medium_stats["coverage_quality"] = (
            weights["count"] * medium_stats["C_count"] +
            weights["spread"] * medium_stats["C_spread"] +
            weights["receiver"] * medium_stats["C_receiver"]
        )

    all_results.append(medium_stats)

    # ==================================================
    # SMALL LEVEL
    # ==================================================
    small_stats = (
        df.groupby(["frame_time", "cell_id_small"])
        .agg(
            n_ipp=("cell_id_small", "count"),
            n_receivers=(station_col, "nunique")
        )
        .reset_index()
    )

    if use_satellite:
        small_sat = (
            df.groupby(["frame_time", "cell_id_small"])[svid_col]
            .nunique()
            .reset_index(name="n_satellites")
        )

        small_stats = small_stats.merge(
            small_sat,
            on=["frame_time", "cell_id_small"],
            how="left"
        )

    small_stats["grid_level"] = "small"
    small_stats["cell_id"] = small_stats["cell_id_small"].astype(str)

    small_stats["C_count"] = small_stats["n_ipp"].apply(
        lambda n: normalise_count(n, n_good_small)
    )

    # No lower grid level exists below small in the current hierarchy.
    # Therefore, spread is not evaluated internally for small cells.
    small_stats["C_spread"] = 1.0

    small_stats["C_receiver"] = small_stats["n_receivers"].apply(normalise_receiver)

    if use_satellite:
        small_stats["C_satellite"] = small_stats["n_satellites"].apply(normalise_satellite)

        small_stats["coverage_quality"] = (
            weights["count"] * small_stats["C_count"] +
            weights["spread"] * small_stats["C_spread"] +
            weights["receiver"] * small_stats["C_receiver"] +
            weights["satellite"] * small_stats["C_satellite"]
        )
    else:
        small_stats["coverage_quality"] = (
            weights["count"] * small_stats["C_count"] +
            weights["spread"] * small_stats["C_spread"] +
            weights["receiver"] * small_stats["C_receiver"]
        )

    all_results.append(small_stats)

    # ==================================================
    # Combine results
    # ==================================================
    coverage_df = pd.concat(all_results, ignore_index=True)

    coverage_df.loc[
        coverage_df["n_ipp"] < min_ipp_per_cell,
        "coverage_quality"
    ] = np.nan

    coverage_df["coverage_quality"] = coverage_df["coverage_quality"].clip(0, 1)

    coverage_df["coverage_class"] = coverage_df["coverage_quality"].apply(
        classify_coverage_quality
    )

    return coverage_df

def plot_laea_solution2_IT_timeline_multilevel_bokeh_coverage_quality(
    df_ipp,
    coverage_df,
    cell_size,
    lat_south=50,
    projection="3574",
    initial_grid_level="medium",
    lon_col="ipp_sp3_lon",
    lat_col="ipp_sp3_lat",
    time_col="UTC Time",
    it_value_col="I_T",
    time_freq="min",
    time_start=None,
    time_end=None,
    min_ipp_per_cell=4,
    width=750,
    height=750,
    grid_line_color="black",
    grid_line_width=0.8,
    grid_line_alpha=0.8,
    no_data_alpha=0.03,
    ipp_size=7,
    ipp_alpha=0.9,
    show_underlay=True,
    coastline_scale="50m",
    csv_path_swado=None,
    csv_path_chain=None,
    station_lon_col="longitude",
    station_lat_col="latitude",
    station_name_col="station",
    show_north_pole=True,
):
    """
    Bokeh LAEA timeline plot with selectable grid level and selectable cell statistic.

    Cell colour:
        Selected AIMS I_T statistic:
            mean, max, or median

    Cell transparency:
        Coverage quality class:
            Poor, Moderate, Good

    IPP colour:
        Individual AIMS I_T class.
    """

    if initial_grid_level not in ["large", "medium", "small"]:
        raise ValueError("initial_grid_level must be 'large', 'medium', or 'small'")

    # --------------------------------------------------
    # Projection and grid setup
    # --------------------------------------------------
    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    grid_large, grid_medium, grid_small = build_laea_solution2_hierarchy(
        cell_size=cell_size,
        arctic_radius=arctic_radius
    )

    grids = {
        "large": grid_large.dropna(subset=["polygon"]).copy(),
        "medium": grid_medium.dropna(subset=["polygon"]).copy(),
        "small": grid_small.dropna(subset=["polygon"]).copy()
    }

    cell_cols = {
        "large": "cell_id_large",
        "medium": "cell_id_medium",
        "small": "cell_id_small"
    }

    cell_size_km = {
        "large": cell_size / 1000,
        "medium": cell_size / 2 / 1000,
        "small": cell_size / 4 / 1000
    }

    # --------------------------------------------------
    # Data preparation
    # --------------------------------------------------
    df = df_ipp.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if time_start is not None:
        df = df[df[time_col] >= pd.to_datetime(time_start)].copy()

    if time_end is not None:
        df = df[df[time_col] <= pd.to_datetime(time_end)].copy()

    required_cols = [lon_col, lat_col, time_col, it_value_col, "Elevation (degrees)"]

    for level in ["large", "medium", "small"]:
        required_cols.append(cell_cols[level])

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df_ipp: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    if time_freq == "min":
        df["frame_time"] = df[time_col].dt.floor("min")
    elif time_freq == "5min":
        df["frame_time"] = df[time_col].dt.floor("5min")
    elif time_freq == "10min":
        df["frame_time"] = df[time_col].dt.floor("10min")
    elif time_freq is None:
        df["frame_time"] = df[time_col]
    else:
        raise ValueError("time_freq must be 'min', '5min', '10min', or None")

    frame_times = sorted(df["frame_time"].dropna().unique())

    if len(frame_times) == 0:
        raise ValueError("No valid frame times found.")

    coverage = coverage_df.copy()
    coverage["frame_time"] = pd.to_datetime(coverage["frame_time"])
    coverage["cell_id"] = coverage["cell_id"].astype(str)

    ipp_x, ipp_y = transformer_wgs84_to_laea.transform(
        df[lon_col].values,
        df[lat_col].values
    )

    df["ipp_x_laea"] = ipp_x
    df["ipp_y_laea"] = ipp_y
    df["I_T_class"] = df[it_value_col].apply(classify_IT_value)
    df["I_T_color"] = df["I_T_class"].apply(IT_class_to_color)

    time_labels = [
        pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S UTC")
        for t in frame_times
    ]

    # --------------------------------------------------
    # Precompute frames
    # --------------------------------------------------
    grid_frames_by_level = {}
    ipp_frames_by_level = {}
    frame_summary_by_level = {}

    for level in ["large", "medium", "small"]:
        grid_df = grids[level]
        cell_col = cell_cols[level]

        xs, ys, grid_cell_ids = [], [], []

        for _, row_i in grid_df.iterrows():
            x_poly, y_poly = row_i["polygon"].exterior.xy
            xs.append(list(x_poly))
            ys.append(list(y_poly))
            grid_cell_ids.append(str(row_i["cell_id"]))

        base_grid = pd.DataFrame({
            "cell_id": grid_cell_ids,
            "xs": xs,
            "ys": ys
        })

        grid_frames = {}
        ipp_frames = {}
        frame_summary = {}

        for i, t in enumerate(frame_times):
            df_t = df[df["frame_time"] == t].copy()

            cell_stats = (
                df_t
                .groupby(cell_col)
                .agg(
                    I_T_mean=(it_value_col, "mean"),
                    I_T_median=(it_value_col, "median"),
                    I_T_std=(it_value_col, "std"),
                    I_T_max=(it_value_col, "max"),
                    I_T_min=(it_value_col, "min"),
                    I_T_q25=(it_value_col, lambda x: x.quantile(0.25)),
                    I_T_q75=(it_value_col, lambda x: x.quantile(0.75)),
                    n_ipp=(it_value_col, "count")
                )
                .reset_index()
            )

            cell_stats["I_T_iqr"] = cell_stats["I_T_q75"] - cell_stats["I_T_q25"]
            cell_stats["I_T_range"] = cell_stats["I_T_max"] - cell_stats["I_T_min"]

            # Classification for each possible displayed statistic
            cell_stats["I_T_class_mean"] = cell_stats["I_T_mean"].apply(classify_IT_value)
            cell_stats["I_T_color_mean"] = cell_stats["I_T_class_mean"].apply(IT_class_to_color)

            cell_stats["I_T_class_median"] = cell_stats["I_T_median"].apply(classify_IT_value)
            cell_stats["I_T_color_median"] = cell_stats["I_T_class_median"].apply(IT_class_to_color)

            cell_stats["I_T_class_max"] = cell_stats["I_T_max"].apply(classify_IT_value)
            cell_stats["I_T_color_max"] = cell_stats["I_T_class_max"].apply(IT_class_to_color)

            low_coverage = cell_stats["n_ipp"] < min_ipp_per_cell

            for stat in ["mean", "median", "max"]:
                cell_stats.loc[low_coverage, f"I_T_class_{stat}"] = "Low coverage"
                cell_stats.loc[low_coverage, f"I_T_color_{stat}"] = "white"

            grid_t = base_grid.merge(
                cell_stats,
                left_on="cell_id",
                right_on=cell_col,
                how="left"
            )

            coverage_t = coverage[
                (coverage["grid_level"] == level) &
                (coverage["frame_time"] == pd.to_datetime(t))
            ].copy()

            grid_t = grid_t.merge(
                coverage_t[
                    [
                        "cell_id",
                        "coverage_quality",
                        "coverage_class",
                        "C_count",
                        "C_spread",
                        "C_receiver"
                    ]
                ],
                on="cell_id",
                how="left"
            )

            grid_t["n_ipp"] = grid_t["n_ipp"].fillna(0).astype(int)
            grid_t["coverage_class"] = grid_t["coverage_class"].fillna("No data")

            for stat in ["mean", "median", "max"]:
                grid_t[f"I_T_class_{stat}"] = grid_t[f"I_T_class_{stat}"].fillna("No data")
                grid_t[f"I_T_color_{stat}"] = grid_t[f"I_T_color_{stat}"].fillna("white")

            for c in [
                "I_T_mean",
                "I_T_median",
                "I_T_std",
                "I_T_max",
                "I_T_min",
                "I_T_q25",
                "I_T_q75",
                "I_T_iqr",
                "I_T_range",
                "coverage_quality",
                "C_count",
                "C_spread",
                "C_receiver"
            ]:
                grid_t[c] = grid_t[c].astype(float)

            # Class-based alpha from coverage quality
            grid_t["cell_fill_alpha"] = no_data_alpha

            grid_t.loc[grid_t["coverage_class"] == "Poor", "cell_fill_alpha"] = 0.12
            grid_t.loc[grid_t["coverage_class"] == "Moderate", "cell_fill_alpha"] = 0.38
            grid_t.loc[grid_t["coverage_class"] == "Good", "cell_fill_alpha"] = 0.80

            # Cells with no valid I_T support should remain nearly transparent
            for stat in ["mean", "median", "max"]:
                invalid_stat = grid_t[f"I_T_class_{stat}"].isin(["No data", "Low coverage"])
                grid_t.loc[invalid_stat, f"I_T_color_{stat}"] = "white"

            grid_t.loc[
                grid_t["I_T_class_mean"].isin(["No data", "Low coverage"]),
                "cell_fill_alpha"
            ] = no_data_alpha

            # Initial displayed statistic is mean
            grid_t["I_T_display"] = grid_t["I_T_mean"]
            grid_t["I_T_display_class"] = grid_t["I_T_class_mean"]
            grid_t["I_T_display_color"] = grid_t["I_T_color_mean"]

            coloured_cells = int(
                (~grid_t["I_T_class_mean"].isin(["No data", "Low coverage"])).sum()
            )

            frame_summary[str(i)] = dict(
                active_ipps=int(len(df_t)),
                coloured_cells=coloured_cells
            )

            grid_frames[str(i)] = dict(
                xs=grid_t["xs"].tolist(),
                ys=grid_t["ys"].tolist(),
                cell_id=grid_t["cell_id"].astype(str).tolist(),
                n_ipp=grid_t["n_ipp"].tolist(),

                I_T_mean=grid_t["I_T_mean"].tolist(),
                I_T_median=grid_t["I_T_median"].tolist(),
                I_T_std=grid_t["I_T_std"].tolist(),
                I_T_max=grid_t["I_T_max"].tolist(),
                I_T_min=grid_t["I_T_min"].tolist(),
                I_T_q25=grid_t["I_T_q25"].tolist(),
                I_T_q75=grid_t["I_T_q75"].tolist(),
                I_T_iqr=grid_t["I_T_iqr"].tolist(),
                I_T_range=grid_t["I_T_range"].tolist(),

                I_T_class_mean=grid_t["I_T_class_mean"].astype(str).tolist(),
                I_T_color_mean=grid_t["I_T_color_mean"].astype(str).tolist(),
                I_T_class_median=grid_t["I_T_class_median"].astype(str).tolist(),
                I_T_color_median=grid_t["I_T_color_median"].astype(str).tolist(),
                I_T_class_max=grid_t["I_T_class_max"].astype(str).tolist(),
                I_T_color_max=grid_t["I_T_color_max"].astype(str).tolist(),

                I_T_display=grid_t["I_T_display"].tolist(),
                I_T_display_class=grid_t["I_T_display_class"].astype(str).tolist(),
                I_T_display_color=grid_t["I_T_display_color"].astype(str).tolist(),

                coverage_quality=grid_t["coverage_quality"].tolist(),
                coverage_class=grid_t["coverage_class"].astype(str).tolist(),
                C_count=grid_t["C_count"].tolist(),
                C_spread=grid_t["C_spread"].tolist(),
                C_receiver=grid_t["C_receiver"].tolist(),
                cell_fill_alpha=grid_t["cell_fill_alpha"].tolist()
            )

            ipp_data = {
                "x": df_t["ipp_x_laea"].tolist(),
                "y": df_t["ipp_y_laea"].tolist(),
                "lon": df_t[lon_col].tolist(),
                "lat": df_t[lat_col].tolist(),
                "time": df_t[time_col].astype(str).tolist(),
                it_value_col: df_t[it_value_col].tolist(),
                "I_T_class": df_t["I_T_class"].astype(str).tolist(),
                "I_T_color": df_t["I_T_color"].astype(str).tolist(),
                "cell_id": df_t[cell_col].astype(str).tolist(),
                "Elevation (degrees)": df_t["Elevation (degrees)"].tolist()
            }

            if "SVID" in df_t.columns:
                ipp_data["SVID"] = df_t["SVID"].astype(str).tolist()
            else:
                ipp_data["SVID"] = [""] * len(df_t)

            ipp_frames[str(i)] = ipp_data

        grid_frames_by_level[level] = grid_frames
        ipp_frames_by_level[level] = ipp_frames
        frame_summary_by_level[level] = frame_summary

    # --------------------------------------------------
    # Initial sources
    # --------------------------------------------------
    initial_index = "0"

    grid_source = ColumnDataSource(
        data=grid_frames_by_level[initial_grid_level][initial_index]
    )

    ipp_source = ColumnDataSource(
        data=ipp_frames_by_level[initial_grid_level][initial_index]
    )

    initial_summary = frame_summary_by_level[initial_grid_level][initial_index]

    base_title = (
        f"Grid with I_T timeline and coverage quality "
        f"| Cell size: {cell_size_km[initial_grid_level]:.0f} km "
        f"| min IPP/cell: {min_ipp_per_cell}"
    )

    p = figure(
        width=width,
        height=height,
        title=f"{base_title} | Time: {time_labels[0]}",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    p.xaxis.axis_label = "LAEA x [m]"
    p.yaxis.axis_label = "LAEA y [m]"

    # --------------------------------------------------
    # Station helper
    # --------------------------------------------------
    def add_stations_to_bokeh(p, csv_path, label, marker_color):
        if csv_path is None:
            return None, None

        stations = pd.read_csv(csv_path)

        x_sta, y_sta = transformer_wgs84_to_laea.transform(
            stations[station_lon_col].values,
            stations[station_lat_col].values
        )

        stations["x_laea"] = x_sta
        stations["y_laea"] = y_sta
        stations["network"] = label

        if station_name_col not in stations.columns:
            stations[station_name_col] = label

        source_sta = ColumnDataSource(stations)

        renderer = p.scatter(
            x="x_laea",
            y="y_laea",
            source=source_sta,
            marker="star",
            size=12,
            color=marker_color,
            alpha=0.85,
            line_color=marker_color,
            line_width=1.0,
            legend_label=label
        )

        hover_sta = HoverTool(
            renderers=[renderer],
            tooltips=[
                ("network", "@network"),
                ("station", f"@{{{station_name_col}}}"),
                ("lon", f"@{{{station_lon_col}}}{{0.00}}"),
                ("lat", f"@{{{station_lat_col}}}{{0.00}}")
            ]
        )

        return renderer, hover_sta

    swado_renderer, swado_hover = add_stations_to_bokeh(
        p=p,
        csv_path=csv_path_swado,
        label="SWADO",
        marker_color="purple"
    )

    chain_renderer, chain_hover = add_stations_to_bokeh(
        p=p,
        csv_path=csv_path_chain,
        label="CHAIN",
        marker_color="blue"
    )

    # --------------------------------------------------
    # Underlay
    # --------------------------------------------------
    if show_underlay:
        add_cartopy_feature_to_bokeh_laea(
            p,
            cfeature.COASTLINE.with_scale(coastline_scale),
            transformer_wgs84_to_laea,
            line_color="dimgray",
            line_width=1,
            line_alpha=0.8
        )

        add_cartopy_feature_to_bokeh_laea(
            p,
            cfeature.BORDERS.with_scale(coastline_scale),
            transformer_wgs84_to_laea,
            line_color="gray",
            line_width=0.7,
            line_alpha=0.6
        )

    # --------------------------------------------------
    # Main renderers
    # --------------------------------------------------
    cell_renderer = p.patches(
        xs="xs",
        ys="ys",
        source=grid_source,
        fill_color="I_T_display_color",
        fill_alpha="cell_fill_alpha",
        line_color=grid_line_color,
        line_width=grid_line_width,
        line_alpha=grid_line_alpha
    )

    ipp_renderer = p.scatter(
        x="x",
        y="y",
        source=ipp_source,
        size=ipp_size,
        color="I_T_color",
        alpha=ipp_alpha,
        line_color="black",
        line_width=0.5,
        legend_label="IPP points"
    )

    # --------------------------------------------------
    # North Pole
    # --------------------------------------------------
    north_pole_hover = None

    if show_north_pole:
        x_np, y_np = transformer_wgs84_to_laea.transform(0, 90)

        north_pole_source = ColumnDataSource(data=dict(
            x=[x_np],
            y=[y_np],
            name=["Geographic North Pole"]
        ))

        north_pole_renderer = p.scatter(
            x="x",
            y="y",
            source=north_pole_source,
            marker="asterisk",
            size=18,
            color="black",
            legend_label="North Pole"
        )

        north_pole_hover = HoverTool(
            renderers=[north_pole_renderer],
            tooltips=[
                ("point", "@name"),
                ("lat", "90.00"),
                ("lon", "0.00")
            ]
        )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    # --------------------------------------------------
    # HTML legend
    # --------------------------------------------------
    legend_div = Div(
        width=260,
        text="""
        <div style="font-family:Arial, sans-serif; font-size:13px; line-height:1.35;">
            <h3 style="margin:0 0 8px 0;">Legend</h3>

            <b>Cell colour: selected I<sub>T</sub> statistic</b><br>
            <div style="margin-top:5px;">
                <span style="display:inline-block;width:14px;height:14px;background:green;border:1px solid #333;margin-right:6px;"></span>
                Green: I<sub>T</sub> &lt; 0.35
            </div>
            <div>
                <span style="display:inline-block;width:14px;height:14px;background:yellow;border:1px solid #333;margin-right:6px;"></span>
                Yellow: 0.35 ≤ I<sub>T</sub> &lt; 0.65
            </div>
            <div>
                <span style="display:inline-block;width:14px;height:14px;background:red;border:1px solid #333;margin-right:6px;"></span>
                Red: I<sub>T</sub> ≥ 0.65
            </div>
            <div>
                <span style="display:inline-block;width:14px;height:14px;background:white;border:1px solid #333;margin-right:6px;"></span>
                White: no data / low coverage
            </div>

            <br>
            <b>Cell transparency: coverage quality</b><br>
            <div style="margin-top:5px;">
                <span style="display:inline-block;width:35px;height:14px;background:rgba(0,0,0,0.12);border:1px solid #333;margin-right:6px;"></span>
                Poor coverage
            </div>
            <div>
                <span style="display:inline-block;width:35px;height:14px;background:rgba(0,0,0,0.38);border:1px solid #333;margin-right:6px;"></span>
                Moderate coverage
            </div>
            <div>
                <span style="display:inline-block;width:35px;height:14px;background:rgba(0,0,0,0.80);border:1px solid #333;margin-right:6px;"></span>
                Good coverage
            </div>

            <br>
            <b>Interactive layers</b><br>
            Use the plot legend to hide/show IPP points, stations, and North Pole.
        </div>
        """
    )

    # --------------------------------------------------
    # Status line below plot
    # --------------------------------------------------
    status_div = Div(
        width=width,
        text=(
            f"<b>Time:</b> {time_labels[0]} "
            f"| <b>active IPPs:</b> {initial_summary['active_ipps']} "
            f"| <b>coloured cells:</b> {initial_summary['coloured_cells']}"
        )
    )

    # --------------------------------------------------
    # Hover tools
    # --------------------------------------------------
    cell_hover = HoverTool(
        renderers=[cell_renderer],
        mode="mouse",
        tooltips=[
            ("cell_id", "@cell_id"),
            ("n_ipp", "@n_ipp"),
            ("displayed I_T", "@I_T_display{0.000}"),
            ("displayed class", "@I_T_display_class"),
            ("mean I_T", "@I_T_mean{0.000}"),
            ("median I_T", "@I_T_median{0.000}"),
            ("max I_T", "@I_T_max{0.000}"),
            ("std I_T", "@I_T_std{0.000}"),
            ("Q25 I_T", "@I_T_q25{0.000}"),
            ("Q75 I_T", "@I_T_q75{0.000}"),
            ("IQR I_T", "@I_T_iqr{0.000}"),
            ("range I_T", "@I_T_range{0.000}"),
            ("coverage", "@coverage_quality{0.000}"),
            ("coverage class", "@coverage_class"),
            ("C_count", "@C_count{0.000}"),
            ("C_spread", "@C_spread{0.000}"),
            ("C_receiver", "@C_receiver{0.000}")
        ]
    )

    ipp_hover = HoverTool(
        renderers=[ipp_renderer],
        mode="mouse",
        tooltips=[
            ("time", "@time"),
            ("SVID", "@SVID"),
            ("Elevation", "@{Elevation (degrees)}{0.0}°"),
            ("lon", "@lon{0.00}"),
            ("lat", "@lat{0.00}"),
            (it_value_col, f"@{{{it_value_col}}}{{0.000}}"),
            ("class", "@I_T_class"),
            ("cell_id", "@cell_id")
        ]
    )

    tools_to_add = [cell_hover, ipp_hover]

    if swado_hover is not None:
        tools_to_add.append(swado_hover)

    if chain_hover is not None:
        tools_to_add.append(chain_hover)

    if north_pole_hover is not None:
        tools_to_add.append(north_pole_hover)

    p.add_tools(*tools_to_add)
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------
    slider = Slider(
        start=0,
        end=len(frame_times) - 1,
        value=0,
        step=1,
        title=f"Time: {time_labels[0]}"
    )

    play_button = Button(label="Play", button_type="success", width=80)

    play_callback = CustomJS(
        args=dict(slider=slider, button=play_button),
        code="""
            if (button.label === "Play") {
                button.label = "Pause";

                button._interval = setInterval(function() {
                    if (slider.value < slider.end) {
                        slider.value = slider.value + 1;
                    } else {
                        slider.value = slider.start;
                    }
                }, 500);

            } else {
                button.label = "Play";
                clearInterval(button._interval);
            }
        """
    )

    play_button.js_on_click(play_callback)

    grid_select = Select(
        title="Grid level",
        value=initial_grid_level,
        options=["large", "medium", "small"]
    )

    statistic_select = Select(
        title="Cell I_T statistic",
        value="mean",
        options=["mean", "median", "max"]
    )

    ipp_checkbox = CheckboxGroup(
        labels=["Show IPP points"],
        active=[0]
    )

    ipp_visibility_callback = CustomJS(
        args=dict(ipp_renderer=ipp_renderer, ipp_checkbox=ipp_checkbox),
        code="""
            ipp_renderer.visible = ipp_checkbox.active.includes(0);
        """
    )

    ipp_checkbox.js_on_change("active", ipp_visibility_callback)

    # --------------------------------------------------
    # Main callback
    # --------------------------------------------------
    callback = CustomJS(
        args=dict(
            grid_source=grid_source,
            ipp_source=ipp_source,
            grid_frames_by_level=grid_frames_by_level,
            ipp_frames_by_level=ipp_frames_by_level,
            frame_summary_by_level=frame_summary_by_level,
            time_labels=time_labels,
            plot=p,
            slider=slider,
            grid_select=grid_select,
            statistic_select=statistic_select,
            cell_size_km=cell_size_km,
            min_ipp_per_cell=min_ipp_per_cell,
            status_div=status_div
        ),
        code="""
            const level = grid_select.value;
            const stat = statistic_select.value;
            const i = slider.value.toString();

            const grid_data = grid_frames_by_level[level][i];

            grid_data["I_T_display"] = grid_data["I_T_" + stat];
            grid_data["I_T_display_class"] = grid_data["I_T_class_" + stat];
            grid_data["I_T_display_color"] = grid_data["I_T_color_" + stat];

            grid_source.data = grid_data;
            ipp_source.data = ipp_frames_by_level[level][i];

            grid_source.change.emit();
            ipp_source.change.emit();

            slider.title = "Time: " + time_labels[slider.value];

            const summary = frame_summary_by_level[level][i];

            status_div.text =
                "<b>Time:</b> " + time_labels[slider.value] +
                " | <b>active IPPs:</b> " + summary.active_ipps +
                " | <b>coloured cells:</b> " + summary.coloured_cells;

            const title =
                "Arctic Grid" +
                " | Statistic: " + stat +
                " | Cell size: " + cell_size_km[level].toFixed(0) + " km";

            plot.title.text = title;
        """
    )

    slider.js_on_change("value", callback)
    grid_select.js_on_change("value", callback)
    statistic_select.js_on_change("value", callback)

    layout = column(
        row(grid_select, statistic_select, play_button, ipp_checkbox, slider),
        row(p, legend_div),
        status_div
    )

    return layout, p, slider, grid_select, statistic_select, grid_frames_by_level, ipp_frames_by_level, df




