"""
prep_data_to_csv.py.

Functions to convert GrIS and AIS NetCDF files to CSV format,
and to plot mean mass change time series for both datasets.
Plots mostly for data exploration.

Authors: Tabea Rettelbach, DTU Space
"""

import logging
import secrets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def netcdf_to_csv_gris(netcdf_path: str, output_csv: str) -> None:
    """
    Convert GrIS NetCDF to CSV.

    Include only grid points with at least one non-NaN measurement.

    Parameters
    ----------
    netcdf_path : str
        Path to the input NetCDF file.
    output_csv : str
        Path to the output CSV file.
    """
    ds = xr.open_dataset(netcdf_path)

    # Adjust variable names if needed
    y_dim = ds["y"].to_numpy()
    x_dim = ds["x"].to_numpy()
    times = ds["Year"].to_numpy()  # Should be years or convertible to years
    mb = ds["VMB"].to_numpy()  # Shape: (y, x, year)
    mb_error = ds["VMBer"].to_numpy()  # Shape: (y, x, year)
    lat2d = ds["lat"].to_numpy()  # Shape: (y, x)
    lon2d = ds["lon"].to_numpy()  # Shape: (y, x)

    rows = []
    for i_y in range(len(y_dim)):
        for i_x in range(len(x_dim)):
            # Only add row if there is at least one non-NaN measurement for this grid point
            mb_vals = mb[i_y, i_x, :]
            if not np.all(np.isnan(mb_vals)):
                lat = lat2d[i_y, i_x]
                lon = lon2d[i_y, i_x]
                row = [lat, lon]
                for i_time, _year in enumerate(times):
                    row.append(mb[i_y, i_x, i_time])
                    row.append(mb_error[i_y, i_x, i_time])
                rows.append(row)

    columns = ["lat", "lon"]
    for year in times:
        columns.append(f"{int(year)}_mb")
        columns.append(f"{int(year)}_mb_error")

    converted_df = pd.DataFrame(rows, columns=columns)
    converted_df.to_csv(output_csv, index=False)
    logging.info("Saved CSV to %s", output_csv)


def compute_yearly_ais_mass(
    dm: np.ndarray, area: np.ndarray, years: np.ndarray, y_dim: np.ndarray, x_dim: np.ndarray, unique_years: np.ndarray
) -> np.ndarray:
    """
    Compute yearly mass for each grid cell from AIS data.

    Since GRACE-like data has irregular time steps, aggregate per calendar year.

    Parameters
    ----------
    dm : np.ndarray
        mass change data (time, y, x)
    area : np.ndarray
        area of each gridcell (y, x)
    years : np.ndarray
        integer years for each time step
    y_dim : np.ndarray
        y dimension coordinates
    x_dim : np.ndarray
        x dimension coordinates
    unique_years : np.ndarray
        unique years present in the data

    Returns
    -------
    np.ndarray
        yearly mass for each grid cell (years, y, x)
    """
    yearly_mass = np.full((len(unique_years), len(y_dim), len(x_dim)), np.nan)

    for i, year in enumerate(unique_years):
        idxs = np.where(years == year)[0]
        if len(idxs) == 0:
            continue
        for i_y in range(len(y_dim)):
            for i_x in range(len(x_dim)):
                vals = [dm[i_time, i_y, i_x] * area[i_y, i_x] for i_time in idxs]
                vals = [v for v in vals if not np.isnan(v) and v != 0]
                if vals:
                    yearly_mass[i, i_y, i_x] = np.mean(vals)
    return yearly_mass


def netcdf_to_csv_ais(netcdf_path: str, output_csv: str) -> None:
    """
    Convert AIS NetCDF to CSV.

    Store the *change to previous year* for each grid cell.
    For each grid cell, average all entries per calendar year, then compute year-on-year change.
    Only includes grid points with at least one non-NaN and non-zero change.

    Parameters
    ----------
    netcdf_path : str
        Path to the input NetCDF file.
    output_csv : str
        Path to the output CSV file.
    """
    ds = xr.open_dataset(netcdf_path)
    y_dim = ds["y"].to_numpy()
    x_dim = ds["x"].to_numpy()
    times = ds["time_dec"].to_numpy()  # decimal years
    dm = ds["dm"].to_numpy()  # (time, y, x)
    area = ds["area"].to_numpy()  # (y, x)
    lat2d = ds["lat"].to_numpy()
    lon2d = ds["lon"].to_numpy()

    years = np.floor(times).astype(int)
    unique_years = np.unique(years)

    # Precompute yearly means for each grid cell using the top-level helper
    yearly_mass = compute_yearly_ais_mass(dm, area, years, y_dim, x_dim, unique_years)

    # Compute year-on-year change
    yearly_mass_diff = np.diff(yearly_mass, axis=0)  # (year-1, y, x)
    diff_years = unique_years[1:]

    rows = []
    for i_y in range(len(y_dim)):
        for i_x in range(len(x_dim)):
            mass_changes = yearly_mass_diff[:, i_y, i_x]
            # keep the same selection criteria but simplify the check expression
            valid_mask = ~np.isnan(mass_changes) & (mass_changes != 0)
            if np.any(valid_mask):
                lat = lat2d[i_y, i_x]
                lon = lon2d[i_y, i_x]
                row = [lat, lon]
                for val in mass_changes:
                    row.append(val)
                    row.append(np.abs(val * 0.15))  # assume error of ±15%
                rows.append(row)

    columns = ["lat", "lon"]
    for year in diff_years:
        columns.append(f"{int(year)}_mb")
        columns.append(f"{int(year)}_mb_error")

    converted_df = pd.DataFrame(rows, columns=columns)
    converted_df.to_csv(output_csv, index=False)
    logger.info("Saved CSV to %s", output_csv)


def combine_gris_ais_csv(gris_csv_path: str, ais_csv_path: str, output_csv_path: str) -> None:
    """
    Combine GrIS and AIS CSV files by stacking their rows.

    Parameters
    ----------
    gris_csv_path : str
        Path to the GrIS CSV file.
    ais_csv_path : str
        Path to the AIS CSV file.
    output_csv_path : str
        Path to the output combined CSV file.
    """
    # Read both CSVs
    df_gris = pd.read_csv(gris_csv_path)
    df_ais = pd.read_csv(ais_csv_path)

    # Get all columns except lat/lon
    year_cols_gris = [col for col in df_gris.columns if col not in ["lat", "lon"]]
    year_cols_ais = [col for col in df_ais.columns if col not in ["lat", "lon"]]
    # Union of all year columns, sorted
    all_year_cols = sorted(set(year_cols_gris + year_cols_ais), key=lambda x: (x.split("_")[0], x))

    # Find missing columns for each DataFrame
    missing_gris = [col for col in all_year_cols if col not in df_gris.columns]
    missing_ais = [col for col in all_year_cols if col not in df_ais.columns]

    # Add missing columns all at once (as empty strings)
    if missing_gris:
        df_gris = pd.concat([df_gris, pd.DataFrame("", index=df_gris.index, columns=missing_gris)], axis=1)
    if missing_ais:
        df_ais = pd.concat([df_ais, pd.DataFrame("", index=df_ais.index, columns=missing_ais)], axis=1)

    # Sort columns: lat, lon, then all year columns
    sorted_cols = ["lat", "lon", *all_year_cols]
    df_gris = df_gris[sorted_cols]
    df_ais = df_ais[sorted_cols]

    # Concatenate
    df_combined = pd.concat([df_gris, df_ais], ignore_index=True)
    df_combined.to_csv(output_csv_path, index=False)
    logger.info("Combined CSV saved to %s", output_csv_path)


def plot_random_timeseries_gris_ais(gris_netcdf_path: str, ais_netcdf_path: str) -> None:
    """
    Plot mass-change time series for random grid cells from GrIS and AIS.

    Use the right y-axis for AIS.

    Parameters
    ----------
    gris_netcdf_path : str
        Path to the GrIS NetCDF file.
    ais_netcdf_path : str
        Path to the AIS NetCDF file.
    """
    # --- GrIS ---
    ds_gris = xr.open_dataset(gris_netcdf_path)
    times_gris = ds_gris["Year"].to_numpy()
    mb_gris = ds_gris["VMB"].to_numpy()  # (y, x, year)

    # Find valid grid cells (at least one non-NaN value)
    valid_gris = np.argwhere(~np.all(np.isnan(mb_gris), axis=2))
    i_y_gris, i_x_gris = secrets.choice(valid_gris.tolist())
    mb_vals = mb_gris[i_y_gris, i_x_gris, :]

    # --- AIS ---
    ds_ais = xr.open_dataset(ais_netcdf_path)
    y_dim_ais = ds_ais["y"].to_numpy()
    x_dim_ais = ds_ais["x"].to_numpy()
    times_ais = ds_ais["time_dec"].to_numpy()
    dm_ais = ds_ais["dm"].to_numpy()  # (time, y, x)
    area_ais = ds_ais["area"].to_numpy()  # (y, x)

    # Find valid grid cells (at least one non-zero and non-NaN value)
    valid_ais = []
    for i_y in range(len(y_dim_ais)):
        for i_x in range(len(x_dim_ais)):
            mass_changes = [dm_ais[i_time, i_y, i_x] * area_ais[i_y, i_x] for i_time in range(len(times_ais))]
            if np.any([not np.isnan(val) and val != 0 for val in mass_changes]):
                valid_ais.append((i_y, i_x))
    i_y_ais, i_x_ais = secrets.choice(valid_ais)
    mass_changes = [dm_ais[i_time, i_y_ais, i_x_ais] * area_ais[i_y_ais, i_x_ais] for i_time in range(len(times_ais))]

    # --- Plot with twin y-axis ---
    _, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.set_xlabel("Year")
    ax1.set_ylabel("GrIS Mass change (kg or kg/gridcell)", color=color1)
    (l1,) = ax1.plot(times_gris, mb_vals, label="GrIS random gridcell", marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("AIS Mass change (kg or kg/gridcell)", color=color2)
    (l2,) = ax2.plot(times_ais, mass_changes, label="AIS random gridcell", marker="s", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Random Gridcell Time Series: GrIS vs AIS")
    plt.tight_layout()
    plt.show()


def plot_sum_timeseries_gris_ais(gris_netcdf_path: str, ais_netcdf_path: str) -> None:
    """
    Plot summed mass-change time series for the GrIS and AIS spatial grids.

    Averages AIS data to calendar years before spatial aggregation.

    Parameters
    ----------
    gris_netcdf_path : str
        Path to the GrIS NetCDF file.
    ais_netcdf_path : str
        Path to the AIS NetCDF file.
    """
    # --- GrIS ---
    ds_gris = xr.open_dataset(gris_netcdf_path)
    times_gris = ds_gris["Year"].to_numpy()
    mb_gris = ds_gris["VMB"].to_numpy()  # (y, x, year)
    sum_mb_gris = np.nansum(mb_gris, axis=(0, 1)) / 10**12

    # --- AIS (year-averaging) and differencing ---
    ds_ais = xr.open_dataset(ais_netcdf_path)
    times_ais = ds_ais["time_dec"].to_numpy()
    years_ais = np.floor(times_ais).astype(int)
    unique_years_ais = np.unique(years_ais)
    dm_ais = ds_ais["dm"].to_numpy()  # (time, y, x)
    area_ais = ds_ais["area"].to_numpy()  # (y, x)

    sum_mass_ais_per_year = []
    for year in unique_years_ais:
        idxs = np.where(years_ais == year)[0]
        if len(idxs) == 0:
            sum_mass_ais_per_year.append(np.nan)
            continue
        dm_year = np.nanmean(dm_ais[idxs, :, :], axis=0)  # (y, x)
        mass_year = dm_year * area_ais  # (y, x)
        sum_mass_ais_per_year.append(np.nansum(mass_year) / 10**12)  # sum over grid in Gt

    sum_mass_ais_per_year = np.array(sum_mass_ais_per_year)
    diff_sum_mass_ais = np.diff(sum_mass_ais_per_year)
    diff_years_ais = unique_years_ais[1:]

    # --- Plot on the same y-axis ---
    plt.figure(figsize=(10, 5))
    plt.xlabel("Year")
    plt.ylabel("Sum Mass Change (Gt/yr)")
    plt.plot(times_gris, sum_mb_gris, label="GrIS sum", marker="o", color="tab:blue")
    plt.plot(diff_years_ais, diff_sum_mass_ais, label="AIS sum", marker="s", color="tab:orange")
    plt.axhline(0, color="gray", alpha=0.3, linewidth=2, zorder=0)
    plt.legend(loc="best")
    plt.title("Sum Mass Change (Yearly Avg): GrIS vs AIS")
    plt.tight_layout()
    plt.show()


def plot_mean_timeseries_gris_ais(gris_netcdf_path: str, ais_netcdf_path: str) -> None:
    """
    Plot mean mass-change time series for the GrIS and AIS spatial grids.

    Averages AIS data to calendar years before spatial aggregation.

    Parameters
    ----------
    gris_netcdf_path : str
        Path to the GrIS NetCDF file.
    ais_netcdf_path : str
        Path to the AIS NetCDF file.
    """
    # --- GrIS ---
    ds_gris = xr.open_dataset(gris_netcdf_path)
    times_gris = ds_gris["Year"].to_numpy()
    mb_gris = ds_gris["VMB"].to_numpy()  # (y, x, year)
    mean_mb_gris = np.nanmean(mb_gris, axis=(0, 1)) / 10**12

    # --- AIS (year-averaging) ---
    ds_ais = xr.open_dataset(ais_netcdf_path)
    times_ais = ds_ais["time_dec"].to_numpy()
    years_ais = np.floor(times_ais).astype(int)
    unique_years_ais = np.unique(years_ais)
    dm_ais = ds_ais["dm"].to_numpy()  # (time, y, x)
    area_ais = ds_ais["area"].to_numpy()  # (y, x)

    # For each year, average all time steps in that year for each grid cell
    mean_mass_ais_per_year = []
    for year in unique_years_ais:
        idxs = np.where(years_ais == year)[0]
        if len(idxs) == 0:
            mean_mass_ais_per_year.append(np.nan)
            continue
        # Average dm over all time steps in this year, then multiply by area
        dm_year = np.nanmean(dm_ais[idxs, :, :], axis=0)  # (y, x)
        mass_year = dm_year * area_ais  # (y, x)
        mean_mass_ais_per_year.append(np.nanmean(mass_year) / 10**12)  # mean over grid

    # --- Plot with twin y-axis ---
    _, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean GrIS Mass change Gt", color=color1)
    (l1,) = ax1.plot(times_gris, mean_mb_gris, label="GrIS mean", marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0, color=color1, alpha=0.2, linewidth=2, zorder=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean AIS Mass change Gt", color=color2)
    (l2,) = ax2.plot(unique_years_ais, mean_mass_ais_per_year, label="AIS mean (yearly avg)", marker="s", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(0, color=color2, alpha=0.2, linewidth=2, zorder=0)

    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Mean Gridcell Time Series (Yearly Avg): GrIS vs AIS")
    plt.tight_layout()
    plt.show()


def plot_mean_timeseries_change_gris_ais_from_csv(gris_csv_path: str, ais_csv_path: str) -> None:
    """
    Plot mean mass-change time series from GrIS and AIS CSV files.

    Averages AIS data to calendar years before spatial aggregation.

    Parameters
    ----------
    gris_csv_path : str
        Path to the GrIS CSV file.
    ais_csv_path : str
        Path to the AIS CSV file.
    """
    # --- GrIS ---
    df_gris = pd.read_csv(gris_csv_path)
    # Extract year columns (ending with "_mb" or "_mb_change")
    gris_year_cols = [col for col in df_gris.columns if col.endswith(("_mb", "_mb_change"))]
    gris_years = [int(col.split("_")[0]) for col in gris_year_cols]
    gris_years, gris_year_cols = zip(*sorted(zip(gris_years, gris_year_cols, strict=True)), strict=True)
    gris_vals = df_gris[list(gris_year_cols)].to_numpy()
    mean_gris = np.nanmean(gris_vals, axis=0)

    # --- AIS ---
    df_ais = pd.read_csv(ais_csv_path)
    ais_year_cols = [col for col in df_ais.columns if col.endswith(("_mb", "_mb_change"))]
    ais_years = [int(col.split("_")[0]) for col in ais_year_cols]
    ais_years, ais_year_cols = zip(*sorted(zip(ais_years, ais_year_cols, strict=True)), strict=True)
    ais_vals = df_ais[list(ais_year_cols)].to_numpy()
    mean_ais = np.nanmean(ais_vals, axis=0)

    # --- Plot with twin y-axis ---
    _, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Δ Mean GrIS Mass change (kg/gridcell)", color=color1)
    (l1,) = ax1.plot(gris_years, mean_gris, label="GrIS Δ mean", marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0, color=color1, alpha=0.2, linewidth=2, zorder=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Δ Mean AIS Mass change (kg/gridcell)", color=color2)
    (l2,) = ax2.plot(ais_years, mean_ais, label="AIS Δ mean", marker="s", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(0, color=color2, alpha=0.2, linewidth=2, zorder=0)

    # Combine legends
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Change to Previous Year: Mean GrIS vs AIS (from CSV)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Manually define your paths here and ensure the files exist at these locations before running the script.
    gris_netcdf_path = ".scrapy/GrIS-Annual-RA-VMB-1992-2020.nc"
    gris_output_csv = ".scrapy/GrIS_MB.csv"
    ais_netcdf_path = ".scrapy/GRAVIS-3_GFZOP_0600_AIS_GRID_TUD_0004.nc"
    ais_output_csv = ".scrapy/AIS_MB.csv"
    comb_output_csv = ".scrapy/AIS_and_GrIS_MB.csv"
    # netcdf_to_csv_gris(gris_netcdf_path, gris_output_csv)
    netcdf_to_csv_ais(ais_netcdf_path, ais_output_csv)

    combine_gris_ais_csv(gris_output_csv, ais_output_csv, comb_output_csv)
