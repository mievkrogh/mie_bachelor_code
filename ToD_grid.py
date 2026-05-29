# Import Packages
from __future__ import annotations
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import pymap3d as pm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import CRS, Transformer
from IPython.display import display, HTML
import os
import sys
import imageio.v2 as imageio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from geopy.distance import great_circle
from shapely.geometry import box
from shapely.ops import transform
from shapely.geometry import box, Polygon, Point
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
plt.rcParams["font.family"] = "Times New Roman"

# # --------------------------------------------------
# # General functions
# # --------------------------------------------------
# Read df_val_all from pickle
def read_df_val_pickle(pickle_path):
    """
    Read df_val_all from pickle.
    """
    return pd.read_pickle(pickle_path)

# Read station data and convert to LAEA coordinates
def read_station_data(csv_path, lat_south, transformer_wgs84_to_laea):
    stations_df = pd.read_csv(csv_path)

    required_cols = ["station", "latitude", "longitude"]
    for col in required_cols:
        if col not in stations_df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    stations_df = stations_df[stations_df["latitude"] >= lat_south].copy()

    station_x_laea, station_y_laea = transformer_wgs84_to_laea.transform(
        stations_df["longitude"].values,
        stations_df["latitude"].values
    )

    stations_df["x_laea"] = station_x_laea
    stations_df["y_laea"] = station_y_laea

    return stations_df

# Define LAEA projection and transformers
def define_laea_projection(projection="3574", wgs84=4326):
    if projection == "3574":
        laea_crs = CRS.from_epsg(3574)
        central_longitude = -40
        lat_label_lon = 105

    elif projection == "3575":
        laea_crs = CRS.from_epsg(3575)
        central_longitude = 10
        lat_label_lon = 135


    elif projection == "3574_lon0_0":
        laea_crs = CRS.from_proj4("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
        central_longitude = 0
        lat_label_lon = 135
    

    else:
        raise ValueError("projection must be '3574', '3575', or '3574_lon0_0'")

    wgs84_crs = CRS.from_epsg(wgs84)

    transformer_wgs84_to_laea = Transformer.from_crs(wgs84_crs, laea_crs, always_xy=True)
    transformer_laea_to_wgs84 = Transformer.from_crs(laea_crs, wgs84_crs, always_xy=True)

    map_projection = ccrs.LambertAzimuthalEqualArea(central_longitude=central_longitude,central_latitude=90)

    return (laea_crs,wgs84_crs,transformer_wgs84_to_laea,transformer_laea_to_wgs84, map_projection, lat_label_lon)

# Arctic Map with stations
def plot_arctic_map_with_stations(projection="3574", csv_path_stations=None, lat_south=50, lat_north=90, show=True):
    """
    Plot Arctic map with latitude/longitude lines and station locations.

    Latitude labels follow the latitude lines.
    Longitude labels remain horizontal.
    White label boxes hide graticule lines behind text.

    Parameters
    ----------
    projection : str, default="3574"
        Projection choice:
            "3574"        -> official EPSG:3574
            "3575"        -> official EPSG:3575
            "3574_lon0_0" -> custom LAEA with lon0 = 0
    """

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(projection=projection)

    stations_df = read_station_data(csv_path_stations,lat_south,transformer_wgs84_to_laea)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=map_projection)
    ax.set_extent([-180, 180, lat_south, lat_north], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),edgecolor="dimgray",linewidth=0.8,zorder=2)
    ax.add_feature(cfeature.BORDERS,edgecolor="gray",linewidth=0.5,linestyle=":",zorder=2)

    # Latitude lines
    lon_dense = np.linspace(-180, 180, 721)
    latitudes = [55, 70 , 85]
    for lat in latitudes:
        ax.plot(
            lon_dense,
            np.full_like(lon_dense, lat),
            transform=ccrs.PlateCarree(),
            color="lightgray",
            linewidth=0.6,
            linestyle="--",
            alpha=0.7,
            zorder=3
        )

    # Longitude lines
    longitudes = np.arange(-180, 181, 30)
    lat_line = np.linspace(20, 85, 500)
    for lon in longitudes:
        ax.plot(
            np.full_like(lat_line, lon),
            lat_line,
            transform=ccrs.PlateCarree(),
            color="lightgray",
            linewidth=0.6,
            linestyle="--",
            alpha=0.7,
            zorder=3
        )

    # Latitude labels follow the latitude lines
    lat_label_lon = lat_label_lon
    for lat in latitudes:
        ax.text(
            lat_label_lon,
            lat,
            f"{lat}°N",
            transform=ccrs.PlateCarree(),
            fontsize=9,
            color="gray",
            ha="center",
            va="center",
            rotation=-35,
            rotation_mode="anchor",
            zorder=6,
            bbox=dict(
                boxstyle="square,pad=0.08",
                facecolor="white",
                edgecolor="white",
                alpha=1.0
            )
        )

    # Longitude labels stay horizontal
    label_lat = lat_south + 1.5 
    for lon in longitudes:
        if lon == 0:
            label = "0°"
        elif abs(lon) == 180:
            label = "180°"
        elif lon < 0:
            label = f"{abs(lon)}°W"
        else:
            label = f"{lon}°E"

        ax.text(
            lon,
            label_lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            color="gray",
            ha="center",
            va="center",
            rotation=0,
            zorder=6,
            bbox=dict(
                boxstyle="square,pad=0.08",
                facecolor="white",
                edgecolor="white",
                alpha=1.0
            )
        )


    ax.scatter(
        0, 0,
        marker="+",
        s=100,
        linewidths=2.0,
        c="black",
        transform=map_projection,
        zorder=7,
        label="Geodetic North Pole"
    )

    # ax.set_title(
    #     "Arctic Polar Map",
    #     fontsize=15,
    #     fontweight="bold",
    #     pad=16
    # )

    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)


    # --------------------------------------------------
    # Scale bar with two segments
    # --------------------------------------------------
    scale_length_km =  1000
    scale_length_m = scale_length_km * 1000
    segment_m = scale_length_m / 2

    x0_axes = 0.845
    y0_axes = 0.05

    x_min, x_max = ax.get_xbound()
    y_min, y_max = ax.get_ybound()

    x0 = x_min + x0_axes * (x_max - x_min)
    y0 = y_min + y0_axes * (y_max - y_min)

    bar_height = 0.01 * (y_max - y_min)

    # First segment
    ax.fill(
        [x0, x0 + segment_m, x0 + segment_m, x0],
        [y0, y0, y0 + bar_height, y0 + bar_height],
        color="black",
        transform=map_projection,
        zorder=20)

    # Second segment
    ax.fill(
        [x0 + segment_m, x0 + scale_length_m, x0 + scale_length_m, x0 + segment_m],
        [y0, y0, y0 + bar_height, y0 + bar_height],
        color="white",
        edgecolor="black",
        transform=map_projection,
        zorder=20)

    # Outline
    ax.plot(
        [x0, x0 + scale_length_m, x0 + scale_length_m, x0, x0],
        [y0, y0, y0 + bar_height, y0 + bar_height, y0],
        color="black",
        linewidth=1.2,
        transform=map_projection,
        zorder=21)

    # Labels
    ax.text(
        x0,
        y0 - 1.5 * bar_height,
        "0",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21)
    ax.text(
        x0 + segment_m,
        y0 - 1.5 * bar_height,
        f"{int(scale_length_km/2)}",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21)
    ax.text(
        x0 + scale_length_m,
        y0 - 1.5 * bar_height,
        f"{int(scale_length_km)} km",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21)
    
    # Projection label text
    if projection in ["3574", "3575"]:
        projection_text = f"EPSG:{projection}"
    elif projection == "3574_lon0_0":
        projection_text = "Custom LAEA"
    else:
        projection_text = f"EPSG:{projection}"

    plt.tight_layout()
    
    if show: 
        ax.text(
        0.98, 0.98,
        f"Lambert Azimuthal Equal Area Projection \n{projection_text}",
        ha="right",
        va="top",
        fontsize=12,
        transform=ax.transAxes,
        zorder=21,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.8))
        
        ax.scatter(
        stations_df["x_laea"].values,
        stations_df["y_laea"].values,
        marker="*",
        s=140,
        c="red",
        edgecolors="red",
        linewidths=0.8,
        transform=map_projection,
        zorder=7,
        label="SWADO stations")

        label_offset_x = 50000
        label_offset_y = 30000
        for _, row in stations_df.iterrows():
            ax.text(
                row["x_laea"] + label_offset_x,
                row["y_laea"] + label_offset_y,
                row["station"],
                transform=map_projection,
                fontsize=7,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7
                )
            )

        plt.show()

    return fig, ax, stations_df

# Greenland map with stations
def plot_greenland_map_with_stations(projection="3574",csv_path_stations=None,lat_south=58,show=True):
    """
    Plot zoomed Greenland map with stations.
    Shows Greenland and the Faroe Islands region.

    Parameters
    ----------
    projection : str, default="3574"
        Projection choice:
            "3574"        -> official EPSG:3574
            "3575"        -> official EPSG:3575
            "3574_lon0_0" -> custom LAEA with lon0 = 0

    csv_path_stations : str
        Path to station CSV.

    lat_south : float, default=58
        Minimum latitude for included stations.

    show : bool, default=True
        Whether to show the figure.
    """

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    stations_df = read_station_data(csv_path_stations, lat_south, transformer_wgs84_to_laea)

    # Keep only stations roughly in Greenland + Faroe Islands region
    stations_df = stations_df[
        (stations_df["longitude"] >= -75) &
        (stations_df["longitude"] <= -5) &
        (stations_df["latitude"] >= lat_south) &
        (stations_df["latitude"] <= 85)
    ].copy()

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(1, 1, 1, projection=map_projection)

    # Greenland + Faroe Islands zoom
    ax.set_extent([-75, 0, lat_south, 90], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=1)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="dimgray",
        linewidth=0.8,
        zorder=2
    )
    ax.add_feature(
        cfeature.BORDERS,
        edgecolor="gray",
        linewidth=0.5,
        linestyle=":",
        zorder=2
    )

    ax.scatter(
        0, 0,
        marker="+",
        s=100,
        linewidths=2.0,
        c="black",
        transform=map_projection,
        zorder=7,
        label="Geodetic North Pole"
    )

    # Latitude lines
    lon_dense = np.linspace(0, 360, 500)
    latitudes = [55, 70, 85]
    for lat in latitudes:
        ax.plot(
            lon_dense,
            np.full_like(lon_dense, lat),
            transform=ccrs.PlateCarree(),
            color="lightgray",
            linewidth=0.6,
            linestyle="--",
            alpha=0.7,
            zorder=3
        )

    # Longitude lines
    longitudes = [-120,-90,-60, -30, 0,30]
    lat_line = np.linspace(45, 85, 400)
    for lon in longitudes:
        ax.plot(
            np.full_like(lat_line, lon),
            lat_line,
            transform=ccrs.PlateCarree(),
            color="lightgray",
            linewidth=0.6,
            linestyle="--",
            alpha=0.7,
            zorder=3
        )

    # Latitude labels
    for lat in latitudes:
        ax.text(
            -70,
            lat,
            f"{lat}°N",
            transform=ccrs.PlateCarree(),
            fontsize=9,
            color="gray",
            ha="center",
            va="center",
            rotation=0,
            rotation_mode="anchor",
            zorder=6,
            bbox=dict(
                boxstyle="square,pad=0.08",
                facecolor="white",
                edgecolor="white",
                alpha=1.0
            )
        )

    # Longitude labels
    if lat_south == 55: 
        label_lat = 55
        for lon in longitudes:
            lon_plot=lon
            if lon == -90:
                label_lat = 66
                name=f"{abs(lon)}°W"
            elif lon == 0:
                label_lat = 57
                name=f"{abs(lon)}°W"
            elif lon == 30:
                label_lat = 68
                name=f"{abs(lon)}°E"
            elif lon == -120:
                label_lat = 71
                name=f"{abs(lon)}°W"
            elif lon == -60:   
                label_lat = 54
                name=f"{abs(lon)}°W"
            elif lon == -30:
                label_lat = 55
                name=f"{abs(lon)}°W"
            
            ax.text(
                lon_plot,
                label_lat,
                name,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                color="gray",
                ha="center",
                va="center",
                rotation=0,
                zorder=6,
                bbox=dict(
                    boxstyle="square,pad=0.08",
                    facecolor="white",
                    edgecolor="white",
                    alpha=1.0
                )
            )

    if lat_south == 50:
        label_lat = 55
        for lon in longitudes:
            lon_plot=lon
            if lon == -90:
                label_lat = 63
                name=f"{abs(lon)}°W"
            elif lon == 0:
                label_lat = 53
                name=f"{abs(lon)}°W"
            elif lon == 30:
                label_lat = 65
                name=f"{abs(lon)}°E"
            elif lon == -120:
                label_lat = 69
                name=f"{abs(lon)}°W"
            elif lon == -60:   
                label_lat = 49
                name=f"{abs(lon)}°W"
            elif lon == -30:
                label_lat = 51
                name=f"{abs(lon)}°W"
            
            ax.text(
                lon_plot,
                label_lat,
                name,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                color="gray",
                ha="center",
                va="center",
                rotation=0,
                zorder=6,
                bbox=dict(
                    boxstyle="square,pad=0.08",
                    facecolor="white",
                    edgecolor="white",
                    alpha=1.0
                )
            )

    # Stations in LAEA coordinates
    ax.scatter(
        stations_df["x_laea"].values,
        stations_df["y_laea"].values,
        marker="*",
        s=140,
        c="red",
        edgecolors="red",
        linewidths=0.8,
        transform=map_projection,
        zorder=7,
        label="SWADO stations"
    )

    # Station labels
    label_offset_x = 35000
    label_offset_y = 25000
    for _, row in stations_df.iterrows():
        ax.text(
            row["x_laea"] + label_offset_x,
            row["y_laea"] + label_offset_y,
            row["station"],
            transform=map_projection,
            fontsize=7,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.7
            )
        )

    # ax.set_title(
    #     "Greenland and Faroe Islands Map",
    #     fontsize=15,
    #     fontweight="bold",
    #     pad=16
    # )

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # --------------------------------------------------
    # Scale bar with two segments
    # --------------------------------------------------
    scale_length_km = 500
    scale_length_m = scale_length_km * 1000
    segment_m = scale_length_m / 2

    x0_axes = 0.78
    y0_axes = 0.05

    x_min, x_max = ax.get_xbound()
    y_min, y_max = ax.get_ybound()

    x0 = x_min + x0_axes * (x_max - x_min)
    y0 = y_min + y0_axes * (y_max - y_min)

    bar_height = 0.012 * (y_max - y_min)

    ax.fill(
        [x0, x0 + segment_m, x0 + segment_m, x0],
        [y0, y0, y0 + bar_height, y0 + bar_height],
        color="black",
        transform=map_projection,
        zorder=20
    )

    ax.fill(
        [x0 + segment_m, x0 + scale_length_m, x0 + scale_length_m, x0 + segment_m],
        [y0, y0, y0 + bar_height, y0 + bar_height],
        color="white",
        edgecolor="black",
        transform=map_projection,
        zorder=20
    )

    ax.plot(
        [x0, x0 + scale_length_m, x0 + scale_length_m, x0, x0],
        [y0, y0, y0 + bar_height, y0 + bar_height, y0],
        color="black",
        linewidth=1.2,
        transform=map_projection,
        zorder=21
    )

    ax.text(
        x0,
        y0 - 1.5 * bar_height,
        "0",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21
    )
    ax.text(
        x0 + segment_m,
        y0 - 1.5 * bar_height,
        f"{int(scale_length_km/2)}",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21
    )
    ax.text(
        x0 + scale_length_m,
        y0 - 1.5 * bar_height,
        f"{int(scale_length_km)} km",
        ha="center",
        va="top",
        fontsize=9,
        transform=map_projection,
        zorder=21
    )

    ax.text(
        0.98, 0.98,
        f"Lambert Azimuthal Equal Area Projection\nEPSG:{projection}",
        ha="right",
        va="top",
        fontsize=11,
        transform=ax.transAxes,
        zorder=21,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.8))


    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, stations_df

# # --------------------------------------------------
# # Functions to build LAEA grid
# # --------------------------------------------------
# SOLUTION 1: North Pole at centre of cell (0,0)
def build_laea_solution1(cell_size, arctic_radius):
    N = int(np.ceil(arctic_radius / cell_size)) + 1
    x_coords = (np.arange(-N, N + 1) - 0.5) * cell_size
    y_coords = (np.arange(-N, N + 1) - 0.5) * cell_size

    grid_cells = []
    cell_id_num = 0

    for ix, x in enumerate(x_coords[:-1]):
        for iy, y in enumerate(y_coords[:-1]):
            center_x = x + cell_size / 2
            center_y = y + cell_size / 2
            distance_from_pole = np.sqrt(center_x**2 + center_y**2)

            if distance_from_pole <= arctic_radius:
                ix_rel = int(round(center_x / cell_size))
                iy_rel = int(round(center_y / cell_size))
                cell_id = f"({ix_rel},{iy_rel})"

                grid_cells.append({
                    "cell_id": cell_id,
                    "cell_id_num": cell_id_num,
                    "ix": ix,
                    "iy": iy,
                    "ix_rel": ix_rel,
                    "iy_rel": iy_rel,
                    "center_x": center_x,
                    "center_y": center_y,
                    "polygon": box(x, y, x + cell_size, y + cell_size)
                })
                cell_id_num += 1
    grid_df = pd.DataFrame(grid_cells)
    return grid_df

def plot_arctic_map_with_laea_grid_solution1(csv_path_stations,cell_size,lat_south=50,lat_north=90,grid_cell_label=False,grid_cell_label_fontsize=6,projection="3574"):
    """
    Plot Arctic base map and overlay LAEA grid.
    """
    # Use base function without showing figure yet
    fig, ax, stations_df = plot_arctic_map_with_stations(projection=projection,csv_path_stations=csv_path_stations,lat_south=lat_south,lat_north=lat_north,show=False)

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(projection=projection)

    # Arctic radius
    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    # Build grid
    grid_df = build_laea_solution1(cell_size, arctic_radius)

    # Plot grid
    for _, row in grid_df.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy
        ax.plot(x_cell,y_cell,color="black",linewidth=0.4,alpha=0.6,transform=map_projection,zorder=4)

    # Optional labels
    if grid_cell_label:
        for _, row in grid_df.iterrows():
            ax.text(row["center_x"],row["center_y"],str(row["cell_id"]),transform=map_projection,fontsize=grid_cell_label_fontsize,ha="center",va="center",zorder=8)

    # Update title
    # ax.set_title(
    #     f"Arctic Polar Map",
    #     fontsize=17,
    #     fontweight="bold",
    #     pad=16)

    # Projection label text
    if projection in ["3574", "3575"]:
        projection_text = f"EPSG:{projection}"
    elif projection == "3574_lon0_0":
        projection_text = "Custom LAEA"
    else:
        projection_text = f"EPSG:{projection}"

    ax.text(
        0.98, 0.98,
        f"Lambert Azimuthal Equal Area Projection \n{projection_text} \n  Cell size: {cell_size/1000:.0f} km × {cell_size/1000:.0f} km",
        ha="right",
        va="top",
        fontsize=12,
        transform=ax.transAxes,
        zorder=21,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        )
    )

    plt.show()

    return fig, ax, grid_df, stations_df

# SOLUTION 2: Function to name grid cells
def determine_quadrant_and_local_coords(center_x, center_y):
    """
    Determine quadrant and local positive coordinates.

    Quadrants
    ---------
    1 : upper right  (x > 0, y > 0)
    2 : upper left   (x < 0, y > 0)
    3 : lower left   (x < 0, y < 0)
    4 : lower right  (x > 0, y < 0)

    Returns
    -------
    quadrant : int
    a : float
        Positive local x-distance from origin within quadrant.
    b : float
        Positive local y-distance from origin within quadrant.
    """
    if center_x > 0 and center_y > 0:
        quadrant = 1
        a = center_x
        b = center_y
    elif center_x < 0 and center_y > 0:
        quadrant = 2
        a = -center_x
        b = center_y
    elif center_x < 0 and center_y < 0:
        quadrant = 3
        a = -center_x
        b = -center_y
    elif center_x > 0 and center_y < 0:
        quadrant = 4
        a = center_x
        b = -center_y
    else:
        raise ValueError(
            "Cell centre lies on an axis. "
            "This naming system requires centres to lie strictly inside one quadrant."
        )

    return quadrant, a, b

def determine_large_indices(center_x, center_y, cell_size):
    """
    Determine large-cell indices AA and BB from cell centre.

    Parameters
    ----------
    center_x, center_y : float
        Cell centre coordinates in LAEA [m].
    cell_size : float
        Large-cell size [m].

    Returns
    -------
    quadrant : int
    aa : int
    bb : int
    a : float
    b : float
    """
    quadrant, a, b = determine_quadrant_and_local_coords(center_x, center_y)

    aa = int(np.floor(a / cell_size))
    bb = int(np.floor(b / cell_size))

    if aa > 99 or bb > 99:
        raise ValueError(
            f"AA={aa:02d} or BB={bb:02d} exceeds two-digit limit (00-99)."
        )

    return quadrant, aa, bb, a, b

def determine_medium_subcell(r_a, r_b, cell_size):
    """
    Determine medium-cell index C inside a large cell.

    Large cell is split into 2x2 medium cells.

    C definition
    ------------
    1 : upper right
    2 : upper left
    3 : lower left
    4 : lower right
    """
    half = cell_size / 2

    if r_a >= half and r_b >= half:
        c = 1
    elif r_a < half and r_b >= half:
        c = 2
    elif r_a < half and r_b < half:
        c = 3
    else:
        c = 4

    return c

def determine_small_subcell(r_a, r_b, cell_size):
    """
    Determine small-cell index D inside a medium cell.

    Medium cell is split into 2x2 small cells.

    D definition
    ------------
    1 : upper right
    2 : upper left
    3 : lower left
    4 : lower right
    """
    medium_size = cell_size // 2
    small_size = cell_size // 4

    r_a_mid = r_a % medium_size
    r_b_mid = r_b % medium_size

    if r_a_mid >= small_size and r_b_mid >= small_size:
        d = 1
    elif r_a_mid < small_size and r_b_mid >= small_size:
        d = 2
    elif r_a_mid < small_size and r_b_mid < small_size:
        d = 3
    else:
        d = 4

    return d

def name_solution2_large_cell(center_x, center_y, cell_size):
    """
    Name large cell using format QAABB00.
    """
    quadrant, aa, bb, a, b = determine_large_indices(center_x, center_y, cell_size)
    cell_id = f"{quadrant}{aa:02d}{bb:02d}00"
    return cell_id, quadrant, aa, bb

def name_solution2_medium_cell(center_x, center_y, cell_size):
    """
    Name medium cell using format QAABBC0.
    """
    quadrant, aa, bb, a, b = determine_large_indices(center_x, center_y, cell_size)

    r_a = a - aa * cell_size
    r_b = b - bb * cell_size
    c = determine_medium_subcell(r_a, r_b, cell_size)

    cell_id = f"{quadrant}{aa:02d}{bb:02d}{c}0"
    return cell_id, quadrant, aa, bb, c

def name_solution2_small_cell(center_x, center_y, cell_size):
    """
    Name small cell using format QAABBCD.
    """
    quadrant, aa, bb, a, b = determine_large_indices(center_x, center_y, cell_size)

    r_a = a - aa * cell_size
    r_b = b - bb * cell_size

    c = determine_medium_subcell(r_a, r_b, cell_size)
    d = determine_small_subcell(r_a, r_b, cell_size)

    cell_id = f"{quadrant}{aa:02d}{bb:02d}{c}{d}"
    return cell_id, quadrant, aa, bb, c, d

def build_laea_solution2_hierarchy(cell_size, arctic_radius):
    """
    Build hierarchical LAEA Solution 2 grid in three levels:

    Level 1: large  = cell_size     -> QAABB00
    Level 2: medium = cell_size / 2 -> QAABBC0
    Level 3: small  = cell_size / 4 -> QAABBCD

    The grid is clipped to the Arctic radius using cell-centre distance.

    Parameters
    ----------
    cell_size : float
        Large cell size in metres.
    arctic_radius : float
        Arctic clipping radius in metres.

    Returns
    -------
    grid_large : pandas.DataFrame
    grid_medium : pandas.DataFrame
    grid_small : pandas.DataFrame

    Notes
    -----
    The three returned dataframes are padded with NaN rows at the end
    so they all have the same length.
    """

    # --------------------------------------------------
    # Enforce exact 1:2:4 hierarchy
    # --------------------------------------------------
    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")

    if not isinstance(cell_size, (int, np.integer)):
        raise ValueError(
            "cell_size must be an integer number of metres to enforce an exact 1:2:4 hierarchy."
        )

    if cell_size % 4 != 0:
        raise ValueError(
            f"cell_size={cell_size} does not satisfy the required 1:2:4 hierarchy. "
            "Choose a value divisible by 4."
        )

    large_size = cell_size
    medium_size = cell_size // 2
    small_size = cell_size // 4

    N = int(np.ceil(arctic_radius / large_size)) + 1

    x_large = np.arange(-N * large_size, (N + 1) * large_size, large_size)
    y_large = np.arange(-N * large_size, (N + 1) * large_size, large_size)

    large_cells = []
    medium_cells = []
    small_cells = []

    large_num = 0
    medium_num = 0
    small_num = 0

    for ix_large, x0 in enumerate(x_large[:-1]):
        for iy_large, y0 in enumerate(y_large[:-1]):

            # --------------------------------------------------
            # LARGE CELL
            # --------------------------------------------------
            center_x_large = x0 + large_size / 2
            center_y_large = y0 + large_size / 2

            if not (np.isclose(center_x_large, 0.0) or np.isclose(center_y_large, 0.0)):
                dist_large = np.sqrt(center_x_large**2 + center_y_large**2)

                if dist_large <= arctic_radius:
                    cell_id_large, quadrant, aa, bb = name_solution2_large_cell(
                        center_x_large, center_y_large, large_size
                    )

                    large_cells.append({
                        "cell_id": cell_id_large,
                        "cell_id_num": large_num,
                        "level": "large",
                        "quadrant": quadrant,
                        "AA": aa,
                        "BB": bb,
                        "C": np.nan,
                        "D": np.nan,
                        "center_x": center_x_large,
                        "center_y": center_y_large,
                        "distance_from_pole": dist_large,
                        "polygon": box(x0, y0, x0 + large_size, y0 + large_size)
                    })

                    large_num += 1

            # --------------------------------------------------
            # MEDIUM CELLS inside LARGE
            # --------------------------------------------------
            for i_med in range(2):
                for j_med in range(2):
                    x_med_min = x0 + i_med * medium_size
                    x_med_max = x_med_min + medium_size
                    y_med_min = y0 + j_med * medium_size
                    y_med_max = y_med_min + medium_size

                    center_x_med = x_med_min + medium_size / 2
                    center_y_med = y_med_min + medium_size / 2

                    if np.isclose(center_x_med, 0.0) or np.isclose(center_y_med, 0.0):
                        continue

                    dist_med = np.sqrt(center_x_med**2 + center_y_med**2)

                    if dist_med <= arctic_radius:
                        cell_id_med, quadrant, aa, bb, c = name_solution2_medium_cell(
                            center_x_med, center_y_med, large_size
                        )

                        medium_cells.append({
                            "cell_id": cell_id_med,
                            "cell_id_num": medium_num,
                            "level": "medium",
                            "quadrant": quadrant,
                            "AA": aa,
                            "BB": bb,
                            "C": c,
                            "D": np.nan,
                            "center_x": center_x_med,
                            "center_y": center_y_med,
                            "distance_from_pole": dist_med,
                            "polygon": box(x_med_min, y_med_min, x_med_max, y_med_max)
                        })

                        medium_num += 1

                    # --------------------------------------------------
                    # SMALL CELLS inside MEDIUM
                    # --------------------------------------------------
                    for i_small in range(2):
                        for j_small in range(2):
                            x_small_min = x_med_min + i_small * small_size
                            x_small_max = x_small_min + small_size
                            y_small_min = y_med_min + j_small * small_size
                            y_small_max = y_small_min + small_size

                            center_x_small = x_small_min + small_size / 2
                            center_y_small = y_small_min + small_size / 2

                            if np.isclose(center_x_small, 0.0) or np.isclose(center_y_small, 0.0):
                                continue

                            dist_small = np.sqrt(center_x_small**2 + center_y_small**2)

                            if dist_small <= arctic_radius:
                                cell_id_small, quadrant, aa, bb, c, d = name_solution2_small_cell(
                                    center_x_small, center_y_small, large_size
                                )

                                small_cells.append({
                                    "cell_id": cell_id_small,
                                    "cell_id_num": small_num,
                                    "level": "small",
                                    "quadrant": quadrant,
                                    "AA": aa,
                                    "BB": bb,
                                    "C": c,
                                    "D": d,
                                    "center_x": center_x_small,
                                    "center_y": center_y_small,
                                    "distance_from_pole": dist_small,
                                    "polygon": box(x_small_min, y_small_min, x_small_max, y_small_max)
                                })

                                small_num += 1

    grid_large = pd.DataFrame(large_cells).reset_index(drop=True)
    grid_medium = pd.DataFrame(medium_cells).reset_index(drop=True)
    grid_small = pd.DataFrame(small_cells).reset_index(drop=True)

    # --------------------------------------------------
    # Pad all three dataframes to same length
    # --------------------------------------------------
    max_len = max(len(grid_large), len(grid_medium), len(grid_small))

    def pad_df(df, max_len):
        if len(df) < max_len:
            pad_rows = pd.DataFrame(
                [{col: np.nan for col in df.columns}] * (max_len - len(df))
            )
            df = pd.concat([df, pad_rows], ignore_index=True)
        return df

    grid_large = pad_df(grid_large, max_len)
    grid_medium = pad_df(grid_medium, max_len)
    grid_small = pad_df(grid_small, max_len)

    return grid_large, grid_medium, grid_small

def plot_arctic_map_with_laea_grid_solution2_level(
    csv_path_stations,
    cell_size,
    determine_cell_size="large",
    lat_south=50,
    lat_north=90,
    projection="3574",
    grid_cell_label=False,
    grid_cell_label_fontsize=10,
    show=True
):
    """
    Plot Arctic base map and overlay Solution 2 grid at chosen hierarchy level.

    Parameters
    ----------
    csv_path_stations : str
        Path to station CSV.

    cell_size : float
        Large cell size in metres.

    determine_cell_size : str, default="large"
        Which hierarchy level to plot:
            "large"  -> QAABB00
            "medium" -> QAABBC0
            "small"  -> QAABBCD

    lat_south : float, default=50
    lat_north : float, default=90
    projection : str, default="3574"
    grid_cell_label : bool, default=False
    grid_cell_label_fontsize : int, default=6
    show : bool, default=True

    Returns
    -------
    fig, ax, grid_df_plot, stations_df
    """

    # Base map
    fig, ax, stations_df = plot_arctic_map_with_stations(
        projection=projection,
        csv_path_stations=csv_path_stations,
        lat_south=lat_south,
        lat_north=lat_north,
        show=False
    )

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    # Arctic radius
    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    # Build hierarchy
    grid_large, grid_medium, grid_small = build_laea_solution2_hierarchy(
        cell_size=cell_size,
        arctic_radius=arctic_radius
    )

    # Select level
    if determine_cell_size == "large":
        grid_df_plot = grid_large.dropna(subset=["polygon"]).copy()
        level_name = "Large cells"
        id_format = "QAABB00"
        plotted_cell_size = cell_size

    elif determine_cell_size == "medium":
        grid_df_plot = grid_medium.dropna(subset=["polygon"]).copy()
        level_name = "Medium cells"
        id_format = "QAABBC0"
        plotted_cell_size = cell_size / 2

    elif determine_cell_size == "small":
        grid_df_plot = grid_small.dropna(subset=["polygon"]).copy()
        level_name = "Small cells"
        id_format = "QAABBCD"
        plotted_cell_size = cell_size / 4

    else:
        raise ValueError("determine_cell_size must be 'large', 'medium', or 'small'")

    # Plot grid
    for _, row in grid_df_plot.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy
        ax.plot(
            x_cell,
            y_cell,
            color="black",
            linewidth=0.4,
            alpha=0.6,
            transform=map_projection,
            zorder=4
        )

    # Optional labels
    if grid_cell_label:
        for _, row in grid_df_plot.iterrows():
            ax.text(
                row["center_x"],
                row["center_y"],
                str(row["cell_id"]),
                transform=map_projection,
                fontsize=grid_cell_label_fontsize,
                ha="center",
                va="center",
                color="black",
                zorder=8
            )

    # Projection label text
    if projection in ["3574", "3575"]:
        projection_text = f"EPSG:{projection}"
    elif projection == "3574_lon0_0":
        projection_text = "Custom LAEA"
    else:
        projection_text = str(projection)

    # ax.set_title(
    #     "Arctic Polar Map",
    #     fontsize=17,
    #     fontweight="bold",
    #     pad=16
    # )

    ax.text(
        0.98, 0.98,
        f"Cell size: {plotted_cell_size/1000:.0f} km × {plotted_cell_size/1000:.0f} km\n" 
        f"Level: {level_name}\n"
        f"LAEA Projection\n"
        f"{projection_text}\n",
        # f"Lambert Azimuthal Equal Area Projection\n"
        # f"{projection_text}\n"
        # f"Level: {level_name}\n"
        # f"Plotted cell size: {plotted_cell_size/1000:.0f} km × {plotted_cell_size/1000:.0f} km",
        ha="right",
        va="top",
        fontsize=18,
        transform=ax.transAxes,
        zorder=21,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.6
        )
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_df_plot, stations_df

# PLOT greenland map with grid 
def plot_greenland_map_with_laea_grid_solution2(csv_path_stations,cell_size,lat_south=58,projection="3574",grid_cell_label=False,grid_cell_label_fontsize=6,show=True):
    """
    Plot Greenland + Faroe Islands map and overlay LAEA square grid.

    Parameters
    ----------
    csv_path_stations : str
        Path to station CSV.

    cell_size : float
        Grid cell size in metres.

    lat_south : float, default=58
        Southern latitude boundary used to define Arctic radius.

    projection : str, default="3574"
        Projection choice:
            "3574"        -> official EPSG:3574
            "3575"        -> official EPSG:3575
            "3574_lon0_0" -> custom LAEA with lon0 = 0

    grid_cell_label : bool, default=False
        Whether to show grid cell IDs.

    grid_cell_label_fontsize : int, default=6
        Font size for grid cell labels.

    show : bool, default=True
        Whether to show the figure.

    Returns
    -------
    fig, ax, grid_df, stations_df
    """

    def name_laea_solution2_cell(center_x, center_y, cell_size):
        """
        Create cell name for Solution 2.

        Current naming:
        North Pole is located at the corner of the four central cells.
        Cell IDs are based on the signed index of the cell centre.

        This function is intentionally isolated so the naming scheme
        can easily be changed later.
        """
        ix_rel = int(np.sign(center_x) * np.ceil(abs(center_x) / cell_size))
        iy_rel = int(np.sign(center_y) * np.ceil(abs(center_y) / cell_size))
        cell_id = f"({ix_rel},{iy_rel})"

        return cell_id, ix_rel, iy_rel

    def build_laea_solution2_full(cell_size, arctic_radius):
        """
        Build full square LAEA grid for Solution 2, without radius clipping.

        The extent is still controlled by arctic_radius, in the sense that
        the square grid is built large enough to cover the Arctic domain.
        But no cells are removed based on distance from the pole.
        """
        N = int(np.ceil(arctic_radius / cell_size)) + 1
        x_coords = np.arange(-N * cell_size, (N + 1) * cell_size, cell_size)
        y_coords = np.arange(-N * cell_size, (N + 1) * cell_size, cell_size)

        grid_cells = []
        cell_id_num = 0

        for ix, x in enumerate(x_coords[:-1]):
            for iy, y in enumerate(y_coords[:-1]):
                center_x = x + cell_size / 2
                center_y = y + cell_size / 2

                cell_id, ix_rel, iy_rel = name_laea_solution2_cell(
                    center_x=center_x,
                    center_y=center_y,
                    cell_size=cell_size
                )

                grid_cells.append({
                    "cell_id": cell_id,
                    "cell_id_num": cell_id_num,
                    "ix": ix,
                    "iy": iy,
                    "ix_rel": ix_rel,
                    "iy_rel": iy_rel,
                    "center_x": center_x,
                    "center_y": center_y,
                    "distance_from_pole": np.sqrt(center_x**2 + center_y**2),
                    "polygon": box(x, y, x + cell_size, y + cell_size)
                })

                cell_id_num += 1

        grid_df = pd.DataFrame(grid_cells)
        return grid_df

    def clip_laea_grid_to_arctic_radius(grid_df, arctic_radius):
        """
        Clip grid to Arctic radius based on distance from pole
        computed from cell centre.
        """
        grid_df_clipped = grid_df[grid_df["distance_from_pole"] <= arctic_radius].copy()
        grid_df_clipped = grid_df_clipped.reset_index(drop=True)

        return grid_df_clipped

    def build_laea_solution2(cell_size, arctic_radius):
        """
        Build Arctic-clipped LAEA grid for Solution 2.

        This is the function to use in plotting and analysis when the
        Arctic domain is required.
        """
        grid_df_full = build_laea_solution2_full(cell_size, arctic_radius)
        grid_df_clipped = clip_laea_grid_to_arctic_radius(grid_df_full, arctic_radius)

        return grid_df_clipped


    # Base Greenland map without showing yet
    fig, ax, stations_df = plot_greenland_map_with_stations(
        projection=projection,
        csv_path_stations=csv_path_stations,
        lat_south=lat_south,
        show=False
    )

    laea_crs, wgs84, transformer_wgs84_to_laea, transformer_laea_to_wgs84, map_projection, lat_label_lon = define_laea_projection(
        projection=projection
    )

    # Arctic radius from southern latitude boundary
    x_boundary, y_boundary = transformer_wgs84_to_laea.transform(0, lat_south)
    arctic_radius = np.sqrt(x_boundary**2 + y_boundary**2)

    # Build clipped Arctic grid
    grid_df = build_laea_solution2(cell_size, arctic_radius)

    # Plot grid
    for _, row in grid_df.iterrows():
        x_cell, y_cell = row["polygon"].exterior.xy
        ax.plot(
            x_cell,
            y_cell,
            color="black",
            linewidth=0.4,
            alpha=0.5,
            transform=map_projection,
            zorder=4
        )

    # Optional grid cell labels
    if grid_cell_label:
        for _, row in grid_df.iterrows():
            ax.text(
                row["center_x"],
                row["center_y"],
                str(row["cell_id"]),
                transform=map_projection,
                fontsize=grid_cell_label_fontsize,
                ha="center",
                va="center",
                color="black",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.6
                )
            )

    # Projection label text
    if projection in ["3574", "3575"]:
        projection_text = f"Lambert Azimuthal Equal Area Projection\nEPSG:{projection}"
    elif projection == "3574_lon0_0":
        projection_text = "Lambert Azimuthal Equal Area Projection\nCustom LAEA (lon0 = 0)"
    else:
        projection_text = f"Lambert Azimuthal Equal Area Projection\n{projection}"

    # Update title
    # ax.set_title(
    #     "Greenland and Faroe Islands Map",
    #     fontsize=15,
    #     fontweight="bold",
    #     pad=16
    # )

    # Replace / overwrite projection info box
    ax.text(
        0.98, 0.98,
        f"{projection_text}\nCell size: {cell_size/1000:.0f} km × {cell_size/1000:.0f} km",
        ha="right",
        va="top",
        fontsize=11,
        transform=ax.transAxes,
        zorder=21,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        )
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, grid_df, stations_df

# # --------------------------------------------------
# # Add IPP points to grid cells
# # --------------------------------------------------
def plot_ipp_points(ax, df_val, lon_col="ipp_sp3_lon", lat_col="ipp_sp3_lat",color="blue", size=10, alpha=0.7, label="IPP"):
    """
    Plot IPP points from geodetic lon/lat columns in df_val.
    """
    df_plot = df_val.dropna(subset=[lon_col, lat_col]).copy()

    ax.scatter(
        df_plot[lon_col].values,
        df_plot[lat_col].values,
        s=size,
        c=color,
        alpha=alpha,
        marker="o",
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=5.5,
        label=label
    )

