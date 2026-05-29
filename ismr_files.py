

import pandas as pd

def load_ismr(paths):
    """
    Load multiple ISMR CSV files and return one combined DataFrame.
    
    Parameters
    ----------
    paths : list of str
        List of file paths to ISMR files.
    
    Returns
    -------
    df : pandas.DataFrame
        Combined and processed DataFrame.
    """
    
    # Read all files
    dfs = [pd.read_csv(p, na_values="nan", header=None) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    
    # Assign column names
    df.columns = (
        ["GPS Week Number", "GPS Time of Week", "SVID", "Value of RxState",
        "Azimuth (degrees)", "Elevation (degrees)", 
        "Avg Sig1 (dB-Hz)", "Total s4 on Sig1"]  # Col 1–8

        + ["nan"] * 5                             # Col 9–13
        + ["Phi60"]                               # Col 14

        + ["nan"] * 8                             # Col 15–22
        + ["TEC", "dTEC"]                         # Col 23–24

        + ["nan"] * (62 - 24)                     # Resten
    )
    
    # Convert GPS time to UTC
    gps_seconds = df["GPS Week Number"] * 604800 + df["GPS Time of Week"]
    df["UTC Time"] = pd.to_datetime(
        gps_seconds - 18, 
        unit="s", 
        origin="1980-01-06"
    )
    
    # Convert Phi60 to float
    df["Phi60"] = pd.to_numeric(df["Phi60"], errors="coerce")
    
    return df