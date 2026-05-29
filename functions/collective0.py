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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from geopy.distance import great_circle
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------
# Import Functions
# ---------------------------------------------------------------------
from get_stations import get_station_coordinates, get_stations, stations_geodetic2ecef, station1_geodetic2ecef, extract_array
from sp3_reader import read_sp3, read_sp3_time_window, add_svid_to_sp3
from sp3_spline import interpolate_sp3
from compute_ipp import los_from_ismr, los_from_sp3, compute_ipp, ipp_to_geodetic
from ismr_files import load_ismr


# ---------------------------------------------------------------------
# Build validation DataFrame with IPP coordinates from both pipelines
# ---------------------------------------------------------------------
def build_ipp_validation_df(df_merge: pd.DataFrame,
                            ipp_ecef_sp3: np.ndarray,
                            ipp_ecef_ismr: np.ndarray) -> pd.DataFrame:
    """
    Create validation DataFrame with UTC Time, SVID,
    ECEF IPP coordinates and geodetic IPP coordinates
    from SP3- and ISMR-based pipelines.

    Assumes IPP arrays are aligned 1:1 with df_merge rows.
    """
    
    if "UTC Time" not in df_merge.columns or "SVID" not in df_merge.columns:
        raise ValueError('df_merge must contain columns "UTC Time" and "SVID".')

    a = np.asarray(ipp_ecef_sp3, dtype=float)
    b = np.asarray(ipp_ecef_ismr, dtype=float)

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("ipp_ecef_sp3 must have shape (N, 3).")
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("ipp_ecef_ismr must have shape (N, 3).")

    n = len(df_merge)
    if a.shape[0] != n or b.shape[0] != n:
        raise ValueError(
            f"Length mismatch: len(df_merge)={n}, "
            f"ipp_ecef_sp3={a.shape[0]}, ipp_ecef_ismr={b.shape[0]}."
        )

    # Convert to geodetic
    ipp_geodetic_sp3 = ipp_to_geodetic(a)
    ipp_geodetic_ismr = ipp_to_geodetic(b)

    if ipp_geodetic_sp3.shape != (n, 3):
        raise ValueError("ipp_to_geodetic(sp3) must return shape (N, 3).")
    if ipp_geodetic_ismr.shape != (n, 3):
        raise ValueError("ipp_to_geodetic(ismr) must return shape (N, 3).")
    
    optional_cols = [col for col in ["Total s4 on Sig1", "Phi60", "TEC", "dTEC"] if col in df_merge.columns]
    df_val = df_merge.loc[:, ["UTC Time", "SVID", "Elevation (degrees)"] + optional_cols].copy()

    # ECEF columns
    df_val[["ipp_sp3_x", "ipp_sp3_y", "ipp_sp3_z"]] = a
    df_val[["ipp_ismr_x", "ipp_ismr_y", "ipp_ismr_z"]] = b

    # Geodetic columns
    df_val[["ipp_sp3_lat", "ipp_sp3_lon", "ipp_sp3_h"]] = ipp_geodetic_sp3
    df_val[["ipp_ismr_lat", "ipp_ismr_lon", "ipp_ismr_h"]] = ipp_geodetic_ismr

    df_val = df_val.rename(columns={"Total s4 on Sig1": "S4"})

    return df_val


# ---------------------------------------------------------------------
# Main IPP pipeline for One Station
# ---------------------------------------------------------------------
def ipp_pipeline_one_station(elevation_mask, R_E, h_ion, path_stations, one_station, 
                             paths_ismr, path_sp3, use_spline=True):
    # Stations
    lat, lon, h = get_station_coordinates(path_stations, one_station)
    r_r = station1_geodetic2ecef(path_stations, one_station)

    # load ISMR data
    df_ismr = load_ismr(paths_ismr)
    df_ismr["UTC Time"] = pd.to_datetime(df_ismr["UTC Time"])
    df_ismr = df_ismr[df_ismr["Elevation (degrees)"] > elevation_mask].copy()
    ismr_keys = df_ismr[["SVID", "UTC Time"]].drop_duplicates().copy()

    # Load sp3 files 
    df_sp3 = read_sp3(path_sp3)                          
    df_sp3 = add_svid_to_sp3(df_sp3)                    
    df_sp3 = df_sp3.rename(columns={"epoch_utc": "UTC Time"})
    df_sp3["UTC Time"] = pd.to_datetime(df_sp3["UTC Time"])

    # Spline interpolation of sp3 to ISMR epochs
    if use_spline:
        target_times = df_ismr["UTC Time"].drop_duplicates().sort_values()
        df_sp3 = interpolate_sp3(df_sp3=df_sp3, target_times=target_times, group_col="SVID")

    # Merge ISMR and SP3 data on SVID and UTC Time
    if "epoch_utc" in df_sp3.columns:
        df_sp3 = df_sp3.rename(columns={"epoch_utc": "UTC Time"})
    df_merge = pd.merge(df_ismr, df_sp3, on=["SVID", "UTC Time"], how="inner")

    # Compute IPP coordinates from both pipelines
    # 1. SP3-based pipeline
    array_df_sp3 = extract_array(df_merge)
    los_vec_sp3, los_hat_sp3, rho_sp3 = los_from_sp3(r_station_ecef=r_r, r_sat_ecef=array_df_sp3)
    ipp_ecef_sp3, alpha_sp3 = compute_ipp(r_r_ecef=r_r, los_hat_ecef=los_hat_sp3, h_ion=h_ion, R_E=R_E)

    # 2. ISMR-based pipeline
    los_hat_ismr, df_used = los_from_ismr(
        df_ismr=df_merge,
        station_lat_deg=lat,
        station_lon_deg=lon,
        elevation_min_deg=elevation_mask,
    )
    ipp_ecef_ismr, alpha_ismr = compute_ipp(r_r_ecef=r_r, los_hat_ecef=los_hat_ismr, h_ion=h_ion, R_E=R_E)

    # Build validation DataFrame 
    df_val = build_ipp_validation_df(df_merge, ipp_ecef_sp3, ipp_ecef_ismr)

    return df_val, lat, lon


def generate_ismr_paths_for_station(base_dir, station_code, day_code="019", year_code="26", suffixes=("00", "15", "30")):
    """
    Generate the three ISMR file paths for one station.

    """
    paths = []

    for s in suffixes:
        name = f"{station_code}{day_code}t{s}.{year_code}_.ismr"
        full_path = os.path.join(base_dir, name, name)
        paths.append(full_path)

    return paths

def generate_ismr_paths_by_station(base_dir, station_codes, day_code="019", year_code="26", suffixes=("00", "15", "30")):
    """
    Generate dictionary:
    station_code -> list of ISMR paths
    """
    return {
        station: generate_ismr_paths_for_station(
            base_dir=base_dir,
            station_code=station,
            day_code=day_code,
            year_code=year_code,
            suffixes=suffixes,
        )
        for station in station_codes
    }

# ---------------------------------------------------------------------
# Main IPP pipeline for Multiple Stations
# ---------------------------------------------------------------------
def ipp_pipeline_multiple_stations(elevation_mask, R_E, h_ion, path_stations,
                                   stations_to_run, paths_ismr_by_station,
                                   path_sp3, use_spline=True):
    """
    Run the one-station IPP pipeline for multiple stations and concatenate results.

    Parameters
    ----------
    elevation_mask : float
    R_E : float
    h_ion : float
    path_stations : str
    stations_to_run : list[str]
        List of station names, e.g. ["KULU", "NUUK", "KLQ2"]
    paths_ismr_by_station : dict
        Dictionary mapping station name -> list of ISMR file paths
    path_sp3 : str
    use_spline : bool, default True

    Returns
    -------
    df_val_all : pd.DataFrame
        Concatenated validation dataframe for all stations
    station_info : pd.DataFrame
        Small dataframe with station name and coordinates
    """
    df_list = []
    station_rows = []

    for one_station in stations_to_run:
        if one_station not in paths_ismr_by_station:
            raise ValueError(f"No ISMR paths provided for station '{one_station}'.")

        df_val, lat, lon = ipp_pipeline_one_station(
            elevation_mask=elevation_mask,
            R_E=R_E,
            h_ion=h_ion,
            path_stations=path_stations,
            one_station=one_station,
            paths_ismr=paths_ismr_by_station[one_station],
            path_sp3=path_sp3,
            use_spline=use_spline,
        )

        df_val["Station"] = one_station
        df_list.append(df_val)

        station_rows.append({
            "Station": one_station,
            "Latitude": lat,
            "Longitude": lon,
        })

    df_val_all = pd.concat(df_list, ignore_index=True)
    station_info = pd.DataFrame(station_rows)

    return df_val_all, station_info
