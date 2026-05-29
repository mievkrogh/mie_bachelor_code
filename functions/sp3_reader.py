import numpy as np
import pandas as pd


def read_sp3(
    path: str,
    gps_utc_leap_seconds: int = 18,
) -> pd.DataFrame:
    """
    Read an SP3 file and return a pandas DataFrame.

    Output columns (one row per satellite record per epoch):
      - epoch_gps (datetime): epoch as written in SP3 (assumed GPST)
      - epoch_utc (datetime): epoch_gps converted to UTC (epoch_gps - gps_utc_leap_seconds)
      - record_type ('P' or 'V')
      - sat (e.g., 'G01', 'R02', 'E19')
      - X_ECEF [m], Y_ECEF [m], Z_ECEF [m] (meters)  [for 'P' records]
      - clock_us (microseconds)                      [for 'P' records]
      - raw (original line)

    Notes
    -----
    - Parses epoch lines starting with '*'.
    - Parses satellite records starting with 'P' or 'V'.
    - Positions are stored in SP3 as km -> converted here to meters.
    - Bad/absent clock values (999999.999999) are converted to NaN.
    - Positions that are all zeros are converted to NaN (common absent convention).
    """
    rows = []
    current_epoch_gps = None

    def _safe_float(s: str) -> float:
        s = s.strip()
        if not s:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue

            if line.startswith("*"):
                # *  YYYY MM DD HH MM SS.SSSSSSSS
                year = int(line[3:7])
                month = int(line[8:10])
                day = int(line[11:13])
                hour = int(line[14:16])
                minute = int(line[17:19])
                sec = float(line[20:31])

                sec_int = int(sec)
                micro = int(round((sec - sec_int) * 1_000_000))

                current_epoch_gps = pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                    hour=hour,
                    minute=minute,
                    second=sec_int,
                    microsecond=micro,
                )
                continue

            if line.startswith("EOF"):
                break

            if current_epoch_gps is None:
                # Still in header
                continue

            rec_type = line[0]

            # ---- Position record: P ----
            if rec_type == "P":
                sat = line[1:4].strip()

                # km -> m
                x_km = _safe_float(line[4:18])
                y_km = _safe_float(line[18:32])
                z_km = _safe_float(line[32:46])

                x_m = x_km * 1000.0 if np.isfinite(x_km) else np.nan
                y_m = y_km * 1000.0 if np.isfinite(y_km) else np.nan
                z_m = z_km * 1000.0 if np.isfinite(z_km) else np.nan

                clock_us = _safe_float(line[46:60])

                if np.isfinite(clock_us) and abs(clock_us) >= 999999.0:
                    clock_us = np.nan

                if np.isfinite(x_m) and np.isfinite(y_m) and np.isfinite(z_m):
                    if x_m == 0.0 and y_m == 0.0 and z_m == 0.0:
                        x_m = y_m = z_m = np.nan

                rows.append(
                    {
                        "epoch_gps": current_epoch_gps,
                        "record_type": "P",
                        "sat": sat,
                        "X_ECEF [m]": x_m,
                        "Y_ECEF [m]": y_m,
                        "Z_ECEF [m]": z_m,
                        "clock_us": clock_us,
                        "raw": line.rstrip("\n"),
                    }
                )
                continue

            # ---- Velocity record: V (ignored values kept as NaN here) ----
            if rec_type == "V":
                sat = line[1:4].strip()

                rows.append(
                    {
                        "epoch_gps": current_epoch_gps,
                        "record_type": "V",
                        "sat": sat,
                        "X_ECEF [m]": np.nan,
                        "Y_ECEF [m]": np.nan,
                        "Z_ECEF [m]": np.nan,
                        "clock_us": np.nan,
                        "raw": line.rstrip("\n"),
                    }
                )
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Types
    df["epoch_gps"] = pd.to_datetime(df["epoch_gps"])
    df["sat"] = df["sat"].astype("string")
    for c in ["X_ECEF [m]", "Y_ECEF [m]", "Z_ECEF [m]", "clock_us"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert GPST -> UTC
    df["epoch_utc"] = df["epoch_gps"] - pd.Timedelta(seconds=int(gps_utc_leap_seconds))

    # Reorder so epochs are first
    cols = df.columns.tolist()
    cols.remove("epoch_gps")
    cols.remove("epoch_utc")
    df = df[["epoch_gps", "epoch_utc"] + cols]

    return df.reset_index(drop=True)


def read_sp3_time_window(
    sp3_path: str,
    start_utc: pd.Timestamp | str,
    end_utc: pd.Timestamp | str,
    gps_utc_leap_seconds: int = 18,
) -> pd.DataFrame:
    """
    Read SP3 with read_sp3() and filter by UTC window (inclusive both ends).
    """
    start_utc = pd.Timestamp(start_utc)
    end_utc = pd.Timestamp(end_utc)
    if end_utc < start_utc:
        raise ValueError("end_utc must be >= start_utc")

    df = read_sp3(sp3_path, gps_utc_leap_seconds=gps_utc_leap_seconds)

    mask = (df["epoch_utc"] >= start_utc) & (df["epoch_utc"] <= end_utc)
    return df.loc[mask].copy()


def add_svid_to_sp3(df_sp3: pd.DataFrame, sat_col: str = "sat") -> pd.DataFrame:
    """
    Add Septentrio SVID numbers to SP3 DataFrame based on RINEX satellite codes.

    Parameters
    ----------
    df_sp3 : pd.DataFrame
        SP3 DataFrame containing a satellite column (e.g. 'sat').
    sat_col : str
        Name of satellite column (default 'sat').

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with added column:
        - 'SVID'
    """

    def rinex_sat_to_svid(sat: str) -> int | None:
        if not isinstance(sat, str) or len(sat) < 2:
            return None

        system = sat[0]
        try:
            prn = int(sat[1:])
        except ValueError:
            return None

        if system == "G":  # GPS
            return prn

        if system == "R":  # GLONASS
            if 1 <= prn <= 24:
                return prn + 37
            if 25 <= prn <= 30:
                return prn + 38
            return None

        if system == "E":  # Galileo
            return prn + 70

        if system == "C":  # BeiDou
            if 1 <= prn <= 40:
                return prn + 140
            if 41 <= prn <= 63:
                return prn + 182
            return None

        if system == "J":  # QZSS
            return prn + 180

        if system == "I":  # NavIC
            if 1 <= prn <= 7:
                return prn + 190
            if 8 <= prn <= 14:
                return prn + 208
            return None

        return None

    df = df_sp3.copy()
    df["SVID"] = df[sat_col].map(rinex_sat_to_svid)

    return df
