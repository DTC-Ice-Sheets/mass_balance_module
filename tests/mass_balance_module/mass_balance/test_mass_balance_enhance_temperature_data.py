"""Functions for replacing mass balance temperature data with improved data."""

from unittest.mock import patch

import numpy as np
import xarray as xr

from mass_balance_module.mass_balance import mass_balance_enhance_temperature_data
from mass_balance_module.mass_balance.mass_balance_enhance_temperature_data import (
    TEMPERATURE_KEYWORDS,
    replace_temperature_data,
)


def test_aggregate_year_by_mean():
    data = xr.Dataset(
        {
            "temperature": (("time", "x"), [[1, 2], [3, 4], [5, 6], [7, 8]]),
        },
        coords={
            "time": np.array(["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"], dtype="datetime64"),
            "x": [0, 1],
        },
    )

    aggregated = mass_balance_enhance_temperature_data.aggregate_year_by_mean(data)

    assert set(aggregated.dims) == {"time", "x"}
    assert len(aggregated.time) == 2
    assert np.all(aggregated.time[:] == np.array([2020, 2021]))
    assert len(aggregated.x) == 2
    assert np.array_equal(aggregated["time"].values, [2020, 2021])
    assert np.array_equal(aggregated["temperature"].values, [[2, 3], [6, 7]])


def test_replace_temperature_data():
    base_data = xr.Dataset()
    new_temperature_data = xr.Dataset()
    replaced_data = xr.Dataset({"temperature": (("x",), [1, 2])})

    with patch.object(
        mass_balance_enhance_temperature_data, "replace_feature_data", return_value=replaced_data
    ) as mock_replace_feature_data:
        result = replace_temperature_data(base_data, new_temperature_data)

    mock_replace_feature_data.assert_called_once_with(base_data, new_temperature_data, TEMPERATURE_KEYWORDS)
    assert result is replaced_data
