"""Mass balance enhance temperature data module."""

import xarray as xr

from mass_balance_module.mass_balance.mass_balance_enhance_shared import replace_feature_data

TEMPERATURE_KEYWORDS = ("temp", "2t", "t2m", "167", "lst")


def aggregate_year_by_mean(data: xr.Dataset) -> xr.Dataset:
    """Aggregate along the time dimension taking the mean value per year.

    The time axis will change from datetime to integer year.

    Parameters
    ----------
    data : xr.Dataset
        The input data, with a time dimension.

    Returns
    -------
    xr.Dataset
        The input data, but with the time dimension aggregated to mean values per year.
    """
    return data.groupby("time.year").mean(dim="time").rename({"year": "time"})


def replace_temperature_data(base_data: xr.Dataset, new_temperature_data: xr.Dataset) -> xr.Dataset:
    """Replace the temperature variable in `base_data` with the enhanced temperature data from `new_temperature_data`.

    The enhanced temperature data will be linearly interpolated to the base data's spatial coordinates.

    Parameters
    ----------
    base_data : xr.Dataset
        The original dataset containing the temperature variable to be replaced.
    new_temperature_data : xr.Dataset
        The dataset containing the enhanced temperature variable.

    Returns
    -------
    xr.Dataset
        Modified `base_data` with new temperature data.
    """
    return replace_feature_data(base_data, new_temperature_data, TEMPERATURE_KEYWORDS)
