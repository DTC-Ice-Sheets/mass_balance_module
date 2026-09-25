"""Functions for replacing mass balance elevation-change data with per-year data."""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from use_case_2.mass_balance import mass_balance_enhance_elevation_data
from use_case_2.mass_balance.constants import SECONDS_PER_YEAR
from use_case_2.mass_balance.mass_balance_enhance_elevation_data import (
    ELEVATION_KEYWORDS,
    aggregate_year_by_annual_change,
    replace_elevation_data,
    set_radar_acquisition_features,
)


def test_aggregate_year_by_annual_change():
    # Epochs land close to the year boundaries, so no window needs rescaling.
    times = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01", "2022-01-01"])
    data = xr.Dataset(
        {"dh": (("time", "x"), [[0], [1], [3], [6], [10]])},
        coords={"time": times, "x": [0]},
    )

    aggregated = aggregate_year_by_annual_change(data)

    assert set(aggregated.dims) == {"time", "x"}
    assert np.array_equal(aggregated["time"].values, [2020, 2021])
    # 2020's window [2020-01-01, 2021-01-01] holds epochs 0-2: last (3) minus first (0).
    # 2021's window [2021-01-01, 2022-01-01] holds epochs 2-4: last (10) minus first (3).
    assert np.array_equal(aggregated["dh"].values, [[3], [7]])


def test_aggregate_year_by_annual_change_rescales_partial_window():
    start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-02")
    data = xr.Dataset(
        {"dh": (("time", "x"), [[0], [1]])},
        coords={"time": [start, end], "x": [0]},
    )

    aggregated = aggregate_year_by_annual_change(data)

    span_years = (end - start).total_seconds() / SECONDS_PER_YEAR
    expected_change = (1 - 0) / span_years
    assert np.array_equal(aggregated["time"].values, [2020])
    assert np.allclose(aggregated["dh"].values, [[expected_change]])


def test_aggregate_year_by_annual_change_drops_years_with_one_epoch():
    # 2020 has a single, isolated epoch inside its window, so it must be dropped; 2021 has two.
    times = pd.to_datetime(["2020-06-01", "2021-03-01", "2021-09-01"])
    data = xr.Dataset(
        {"dh": (("time", "x"), [[0], [1], [3]])},
        coords={"time": times, "x": [0]},
    )

    aggregated = aggregate_year_by_annual_change(data)

    assert np.array_equal(aggregated["time"].values, [2021])


def test_aggregate_year_by_annual_change_raises_when_no_year_has_two_epochs():
    data = xr.Dataset(
        {"dh": (("time", "x"), [[0], [1]])},
        coords={"time": pd.to_datetime(["2020-01-01", "2021-06-01"]), "x": [0]},
    )

    with pytest.raises(ValueError, match="No calendar year"):
        aggregate_year_by_annual_change(data)


def test_set_radar_acquisition_features():
    data = xr.Dataset(
        {
            "RAdh": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "RAmode": (("x",), [2.0, 2.0]),
        },
        coords={"time": [2009, 2011], "x": [0, 1]},
    )

    out = set_radar_acquisition_features(data, sarin_start=datetime(2010, 10, 1))

    assert np.array_equal(out["RAdist"].values, [[1.0, 1.0], [1.0, 1.0]])
    # 2009 lies entirely before the SARIn transition, so RAmode is forced to LRM (1.0).
    # 2011 lies after it, so RAmode keeps its baseline value (2.0).
    assert np.array_equal(out["RAmode"].values, [[1.0, 1.0], [2.0, 2.0]])


def test_set_radar_acquisition_features_requires_time_dim():
    data = xr.Dataset({"RAdh": (("x",), [1.0]), "RAmode": (("x",), [2.0])}, coords={"x": [0]})

    with pytest.raises(KeyError, match="time dimension"):
        set_radar_acquisition_features(data)


def test_replace_elevation_data():
    base_data = xr.Dataset()
    new_elevation_data = xr.Dataset()

    replaced_data = xr.Dataset(
        {
            "RAdh": (("time", "y", "x"), [[[15, 25], [35, 45]], [[55, 65], [75, 85]]]),
            "RAmode": (("y", "x"), [[2, 2], [2, 2]]),
        },
        coords={"time": [2009, 2011], "y": [0, 1], "x": [0, 1]},
    )

    with patch.object(
        mass_balance_enhance_elevation_data, "replace_feature_data", return_value=replaced_data
    ) as mock_replace_feature_data:
        result = replace_elevation_data(base_data, new_elevation_data)

    mock_replace_feature_data.assert_called_once_with(base_data, new_elevation_data, ELEVATION_KEYWORDS)

    # set_radar_acquisition_features should have been applied to whatever replace_feature_data returned.
    assert isinstance(result, xr.Dataset)
    assert result["RAdist"].dims == ("time", "y", "x")
    assert np.array_equal(result["RAdist"].values, np.ones((2, 2, 2)))
    assert result["RAmode"].dims == ("time", "y", "x")
    # 2009 is pre-SARIn (forced to LRM), 2011 keeps the baseline value of 2.
    assert np.array_equal(result["RAmode"].values, [[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
