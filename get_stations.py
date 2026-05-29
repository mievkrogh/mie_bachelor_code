import numpy as np
import pandas as pd
import pymap3d as pm

def get_station_coordinates(path: str, station_name: str) -> tuple[float, float, float]:
    """
    Return latitude, longitude and height for a specific station.

    Parameters
    ----------
    path : str
        Path to station CSV file.
    station_name : str
        Name of the station (value in first column).

    Returns
    -------
    lat : float
    lon : float
    height : float

    Raises
    ------
    ValueError
        If station is not found.
    """

    stations = pd.read_csv(path)

    # Assume first column contains station names
    station_column = stations.columns[0]

    row = stations[stations[station_column] == station_name]

    if row.empty:
        raise ValueError(f"Station '{station_name}' not found in file.")

    lat = float(row["latitude"].values[0])
    lon = float(row["longitude"].values[0])
    height = float(row["Height [m]"].values[0])

    return lat, lon, height

def get_stations(path: str) -> pd.DataFrame:
    """
    Read station CSV file and return DataFrame with added ECEF coordinates.

    Parameters
    ----------
    path : str
        Path to station CSV file.

    Returns
    -------
    stations
    """

    stations = pd.read_csv(path)

    lat = stations["latitude"].to_numpy(dtype=float)
    lon = stations["longitude"].to_numpy(dtype=float)
    h   = stations["Height [m]"].to_numpy(dtype=float)
    
    stations["Latitude"] = lat
    stations["Longitude"] = lon
    stations["Height"] = h

    return stations

def stations_geodetic2ecef(path: str) -> pd.DataFrame:
    """
    Read station CSV file and return DataFrame with added ECEF coordinates.

    Parameters
    ----------
    path : str
        Path to station CSV file.

    Returns
    -------
    stations : pd.DataFrame
        DataFrame with added columns:
        - X_ECEF [m]
        - Y_ECEF [m]
        - Z_ECEF [m]
    """

    stations = pd.read_csv(path)

    lat = stations["latitude"].to_numpy(dtype=float)
    lon = stations["longitude"].to_numpy(dtype=float)
    h   = stations["Height [m]"].to_numpy(dtype=float)

    x, y, z = pm.geodetic2ecef(lat, lon, h)

    stations["X_ECEF [m]"] = x
    stations["Y_ECEF [m]"] = y
    stations["Z_ECEF [m]"] = z

    return stations


def station1_geodetic2ecef(path: str, station_name: str) -> np.ndarray:
    """
    Return ECEF receiver vector r_r for a given station.

    Parameters
    ----------
    path : str
        Path to station CSV file.
    station_name : str
        Name of station (e.g., "KLQ2").

    Returns
    -------
    r_r : np.ndarray
        Shape (3,) receiver position in ECEF [m].
    """

    # Reuse the first function
    stations = stations_geodetic2ecef(path)

    row = stations.loc[stations["station"] == station_name]

    if row.empty:
        raise ValueError(f"Station '{station_name}' not found.")

    r_r = row[["X_ECEF [m]", "Y_ECEF [m]", "Z_ECEF [m]"]].values[0]

    return r_r.astype(float)


def extract_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract ECEF coordinates from a DataFrame and return as numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ECEF columns.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns:
        [X_ECEF [m], Y_ECEF [m], Z_ECEF [m]]
    """

    required_cols = ["X_ECEF [m]", "Y_ECEF [m]", "Z_ECEF [m]"]

    # Check that all required columns exist
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required ECEF columns: {missing}")

    # Extract and convert to numpy array
    ecef_array = df[required_cols].to_numpy(dtype=float)

    return ecef_array