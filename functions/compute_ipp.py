import numpy as np
import pandas as pd
import pymap3d as pm


def los_from_ismr(
    df_ismr: pd.DataFrame,
    station_lat_deg: float,
    station_lon_deg: float,
    *,
    az_col: str = "Azimuth (degrees)",
    el_col: str = "Elevation (degrees)",
    elevation_min_deg: float | None = 15.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build LOS unit vectors in ECEF from ISMR azimuth/elevation.

    Parameters
    ----------
    df_ismr : pd.DataFrame
        ISMR dataframe containing azimuth and elevation columns.
    station_lat_deg : float
        Station geodetic latitude [deg].
    station_lon_deg : float
        Station geodetic longitude [deg].
    az_col : str, optional
        Name of azimuth column in df_ismr.
    el_col : str, optional
        Name of elevation column in df_ismr.
    elevation_min_deg : float | None, optional
        Minimum elevation angle [deg]. Rows below this value are removed.
        If None, no filtering is applied.

    Returns
    -------
    los_hat_ecef : (N, 3) ndarray
        Unit LOS vectors in ECEF.
    df_used : pd.DataFrame
        Filtered dataframe used for the LOS computation.
    """
    if az_col not in df_ismr.columns or el_col not in df_ismr.columns:
        raise KeyError(f"Missing columns: {az_col}, {el_col}")

    df_used = df_ismr.copy()

    if elevation_min_deg is not None:
        df_used = df_used[df_used[el_col].astype(float) >= float(elevation_min_deg)].copy()

    if len(df_used) == 0:
        return np.empty((0, 3), dtype=float), df_used

    az = np.deg2rad(df_used[az_col].to_numpy(dtype=float))
    el = np.deg2rad(df_used[el_col].to_numpy(dtype=float))

    # Local LOS in SEZ frame
    # Azimuth convention:
    # 0 deg = North, 90 deg = East, clockwise from North
    s_sez = np.vstack([
        -np.cos(el) * np.cos(az),  # South
         np.cos(el) * np.sin(az),  # East
         np.sin(el)                # Zenith
    ]).T

    phi = np.deg2rad(float(station_lat_deg))
    lam = np.deg2rad(float(station_lon_deg))

    # Vallado-style ECEF -> SEZ rotation
    R_ecef_to_sez = np.array([
        [ np.sin(phi) * np.cos(lam),  np.sin(phi) * np.sin(lam), -np.cos(phi)],
        [-np.sin(lam),                np.cos(lam),                0.0         ],
        [ np.cos(phi) * np.cos(lam),  np.cos(phi) * np.sin(lam),  np.sin(phi)]
    ], dtype=float)

    # Inverse rotation: SEZ -> ECEF
    R_sez_to_ecef = R_ecef_to_sez.T

    los_hat_ecef = (R_sez_to_ecef @ s_sez.T).T
    los_hat_ecef /= np.linalg.norm(los_hat_ecef, axis=1)[:, None]

    return los_hat_ecef, df_used


def los_from_sp3(
    r_station_ecef: np.ndarray,
    r_sat_ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute line-of-sight vectors from station to satellites in ECEF.

    Parameters
    ----------
    r_station_ecef : (3,) array_like
        Station position [m] in ECEF.
    r_sat_ecef : (N, 3) array_like
        Satellite positions [m] in ECEF.

    Returns
    -------
    los_vec : (N, 3) ndarray
        LOS vectors [m].
    los_hat : (N, 3) ndarray
        Unit LOS vectors [-].
    rho : (N,) ndarray
        Slant ranges [m].
    """
    r_station_ecef = np.asarray(r_station_ecef, dtype=float).reshape(3,)
    r_sat_ecef = np.asarray(r_sat_ecef, dtype=float)

    if r_sat_ecef.ndim != 2 or r_sat_ecef.shape[1] != 3:
        raise ValueError("r_sat_ecef must have shape (N, 3).")

    los_vec = r_sat_ecef - r_station_ecef
    rho = np.linalg.norm(los_vec, axis=1)

    if np.any(rho == 0):
        raise ValueError("At least one satellite position equals station position.")

    los_hat = los_vec / rho[:, None]

    return los_vec, los_hat, rho


def compute_ipp(
    r_r_ecef: np.ndarray,
    los_hat_ecef: np.ndarray,
    h_ion: float,
    R_E: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IPP positions in ECEF using the thin-shell model.

    Parameters
    ----------
    r_r_ecef : (3,) array_like
        Receiver position [m] in ECEF.
    los_hat_ecef : (N, 3) array_like
        Unit LOS vectors in ECEF [-].
    h_ion : float
        Ionospheric shell height above Earth's surface [m].
    R_E : float
        Earth radius [m].

    Returns
    -------
    r_ipp : (N, 3) ndarray
        IPP positions [m] in ECEF.
    alpha : (N,) ndarray
        Distance along LOS from receiver to IPP [m].
        NaN where no valid forward intersection exists.
    """
    r_r_ecef = np.asarray(r_r_ecef, dtype=float).reshape(3,)
    los_hat_ecef = np.asarray(los_hat_ecef, dtype=float)

    if los_hat_ecef.ndim != 2 or los_hat_ecef.shape[1] != 3:
        raise ValueError("los_hat_ecef must have shape (N, 3).")

    if los_hat_ecef.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    R_s = float(R_E) + float(h_ion)

    s_dot_r = np.einsum("ij,j->i", los_hat_ecef, r_r_ecef)
    r_r_dot = float(np.dot(r_r_ecef, r_r_ecef))

    discriminant = s_dot_r**2 - (r_r_dot - R_s**2)

    alpha = np.full(los_hat_ecef.shape[0], np.nan, dtype=float)
    ok = discriminant >= 0
    alpha[ok] = -s_dot_r[ok] + np.sqrt(discriminant[ok])

    # Only accept intersections in front of the receiver
    alpha[alpha <= 0] = np.nan

    r_ipp = r_r_ecef + alpha[:, None] * los_hat_ecef

    return r_ipp, alpha


def ipp_to_geodetic(r_ipp_ecef: np.ndarray) -> np.ndarray:
    """
    Convert IPP positions from ECEF to geodetic coordinates.

    Parameters
    ----------
    r_ipp_ecef : (N, 3) array_like
        IPP positions [m] in ECEF.

    Returns
    -------
    ipp_geodetic : (N, 3) ndarray
        Geodetic coordinates [lat_deg, lon_deg, h_m].
    """
    r_ipp_ecef = np.asarray(r_ipp_ecef, dtype=float)

    if r_ipp_ecef.ndim != 2 or r_ipp_ecef.shape[1] != 3:
        raise ValueError("r_ipp_ecef must have shape (N, 3).")

    lat, lon, h = pm.ecef2geodetic(
        r_ipp_ecef[:, 0],
        r_ipp_ecef[:, 1],
        r_ipp_ecef[:, 2],
    )

    ipp_geodetic = np.vstack([lat, lon, h]).T
    return ipp_geodetic
