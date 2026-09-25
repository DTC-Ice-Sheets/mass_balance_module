"""Mass balance enhance elevation data module."""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from mass_balance_module.mass_balance.constants import ANNUAL_WINDOW_TOLERANCE, SARIN_START, SECONDS_PER_YEAR
from mass_balance_module.mass_balance.mass_balance_enhance_shared import find_variable, replace_feature_data

ELEVATION_KEYWORDS = ("dh", "elevation", "sec", "radh")


def aggregate_year_by_annual_change(data: xr.Dataset, tolerance: float = ANNUAL_WINDOW_TOLERANCE) -> xr.Dataset:
    """Aggregate a cumulative elevation-change cube into the change accumulated within each calendar year.

    For every calendar year with at least two epochs inside [1 January, 1 January of the following year], the
    annual value is the last of those epochs' cumulative value minus the first. Epochs rarely land exactly on the
    year boundary, so a window covering a different span than one year is rescaled to its one-year equivalent,
    matching the annual change the model was trained on. Years with fewer than two epochs are dropped.

    The time axis will change from irregularly sampled epochs to one integer year per aggregated value.

    Parameters
    ----------
    data : xr.Dataset
        Cumulative elevation-change cube, with a `time` dimension of irregularly sampled epochs.
    tolerance : float
        Largest fractional deviation from a one-year window tolerated before rescaling, by default
        :data:`ANNUAL_WINDOW_TOLERANCE`.

    Returns
    -------
    xr.Dataset
        The input data, with the time dimension aggregated to one change value per calendar year it can support.

    Raises
    ------
    ValueError
        If no calendar year has at least two epochs to difference.
    """
    var_name = find_variable(data, ELEVATION_KEYWORDS)
    cube = data[var_name]
    times = pd.DatetimeIndex(data["time"].values)

    annual_changes = []
    kept_years = []
    for year in sorted(times.year.unique()):
        start, end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
        inside = (times >= start) & (times <= end)
        if inside.sum() < 2:
            continue

        window_cube = cube.isel(time=np.flatnonzero(inside))
        window_times = times[inside]
        change = window_cube.isel(time=-1) - window_cube.isel(time=0)

        span_years = (window_times[-1] - window_times[0]).total_seconds() / SECONDS_PER_YEAR
        if span_years > 0 and abs(span_years - 1.0) > tolerance:
            change = change / span_years

        annual_changes.append(change)
        kept_years.append(int(year))

    if not annual_changes:
        raise ValueError("No calendar year has at least two epochs of elevation-change data to difference.")

    annual = xr.concat(annual_changes, dim=pd.Index(kept_years, name="time"))
    return annual.to_dataset(name=var_name)


def set_radar_acquisition_features(data: xr.Dataset, sarin_start: datetime = SARIN_START) -> xr.Dataset:
    """Refresh the two radar-altimetry features that co-vary with elevation change, for every year in `data`.

    `RAdist` is held at unity for every point and year, exactly as the training-time feature processor does.
    `RAmode` is set to LRM (1.0) for every year that lies entirely before the CryoSat-2 SARIn transition; later
    years keep their baseline value, which already reflects the SARIn-era acquisition.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with a `time` dimension of calendar years, holding the `RAdh`, `RAdist` and `RAmode` features.
    sarin_start : datetime
        Date CryoSat-2 switched Greenland acquisition from LRM to SARIn, by default :data:`SARIN_START`.

    Returns
    -------
    xr.Dataset
        `data` with `RAdist` and `RAmode` refreshed for every year.

    Raises
    ------
    KeyError
        If `data` carries no time dimension.
    """
    if "time" not in data.dims:
        raise KeyError("`data` has no time dimension to set the per-year radar-acquisition features on.")

    data["RAdist"] = xr.ones_like(data["RAdh"])

    is_lrm_year = xr.DataArray(data["time"].to_numpy() < sarin_start.year, dims="time", coords={"time": data["time"]})
    data["RAmode"] = xr.where(is_lrm_year, 1.0, data["RAmode"])

    return data


def replace_elevation_data(base_data: xr.Dataset, new_elevation_data: xr.Dataset) -> xr.Dataset:
    """Replace the elevation-change feature in `base_data` with the annual elevation-change data.

    The annual elevation-change data will be linearly interpolated to the base data's spatial coordinates. The two
    radar-altimetry features that co-vary with it, `RAdist` and `RAmode`, are refreshed for the same years; every
    other feature in `base_data` keeps its original baseline value across the newly added time dimension.

    Parameters
    ----------
    base_data : xr.Dataset
        The original dataset containing the elevation-change variable to be replaced.
    new_elevation_data : xr.Dataset
        The dataset containing the annual elevation-change data, as returned by :func:`aggregate_year_by_annual_change`.

    Returns
    -------
    xr.Dataset
        Modified `base_data` with new, per-year elevation-change, radar-distance and radar-mode data.
    """
    base_data = replace_feature_data(base_data, new_elevation_data, ELEVATION_KEYWORDS)
    return set_radar_acquisition_features(base_data)
