import numpy as np 
import pandas as pd
from scipy.interpolate import CubicSpline

def interpolate_sp3(
    df_sp3: pd.DataFrame,
    target_times,
    time_col: str = "UTC Time",
    group_col: str = "SVID",
    sat_col: str = "sat",
    x_col: str = "X_ECEF [m]",
    y_col: str = "Y_ECEF [m]",
    z_col: str = "Z_ECEF [m]",
    bc_type: str = "natural",
    min_points: int = 4,
) -> pd.DataFrame:
    """
    Interpolate SP3 satellite ECEF positions with cubic splines, grouped by SVID.
    """
    df = df_sp3.copy()

    # extract time
    df[time_col] = pd.to_datetime(df[time_col])
    target_times = pd.to_datetime(pd.Index(target_times))

    # keep coloumns
    keep_cols = [time_col, group_col, x_col, y_col, z_col]
    if sat_col in df.columns:
        keep_cols.append(sat_col)

    df = df[keep_cols].dropna(subset=[time_col, group_col, x_col, y_col, z_col])

    if df.empty:
        raise ValueError("df_sp3 is empty after dropping NaN values.")

    # choose time
    t0 = min(df[time_col].min(), target_times.min())

    out = []

    # for loop over each satellite group
    for group_value, g in df.groupby(group_col):
        g = g.sort_values(time_col).drop_duplicates(subset=[time_col]).copy()

        # if sat_col is present, get the satellite value for this group
        sat_value = g[sat_col].iloc[0] if sat_col in g.columns else np.nan

        # if there are not enough points for interpolation, return NaN for this group
        if len(g) < min_points:
            result = pd.DataFrame({
                time_col: target_times,
                group_col: group_value,
                x_col: np.nan,
                y_col: np.nan,
                z_col: np.nan,
            })
            if sat_col in df.columns:
                result[sat_col] = sat_value
            out.append(result)
            continue

        t_sec = (g[time_col] - t0).dt.total_seconds().to_numpy()
        target_sec = (target_times - t0).total_seconds()

        xyz = g[[x_col, y_col, z_col]].to_numpy(dtype=float)

        if np.any(np.diff(t_sec) <= 0):
            result = pd.DataFrame({
                time_col: target_times,
                group_col: group_value,
                x_col: np.nan,
                y_col: np.nan,
                z_col: np.nan,
            })
            if sat_col in df.columns:
                result[sat_col] = sat_value
            out.append(result)
            continue


        # perform cubic spline interpolation
        try:
            cs = CubicSpline(
                t_sec,
                xyz,
                axis=0,
                bc_type=bc_type,
                extrapolate=False,
            )

            xyz_interp = cs(target_sec)
            
            # if any of the interpolated values are outside the original time range, set them to NaN
            result = pd.DataFrame({
                time_col: target_times,
                group_col: group_value,
                x_col: xyz_interp[:, 0],
                y_col: xyz_interp[:, 1],
                z_col: xyz_interp[:, 2],
            })
            if sat_col in df.columns:
                result[sat_col] = sat_value

        except Exception:
            result = pd.DataFrame({
                time_col: target_times,
                group_col: group_value,
                x_col: np.nan,
                y_col: np.nan,
                z_col: np.nan,
            })
            if sat_col in df.columns:
                result[sat_col] = sat_value

        out.append(result)

    return pd.concat(out, ignore_index=True)
