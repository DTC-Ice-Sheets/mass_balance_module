"""Tests for the shared mass balance feature-enhancement helpers."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from mass_balance_module.mass_balance import mass_balance_enhance_shared
from mass_balance_module.mass_balance.mass_balance_enhance_shared import (
    find_variable,
    find_x_y_coords,
    interpolate_to_new_coords,
    replace_feature_data,
)


def test_find_variable():
    data = xr.Dataset({"land_surface_temperature": (("x",), [1, 2]), "other_var": (("x",), [3, 4])})

    assert find_variable(data, ["temp", "lst"]) == "land_surface_temperature"


def test_find_variable_not_found():
    data = xr.Dataset({"other_var": (("x",), [3, 4])})

    with pytest.raises(ValueError, match="No variable matching"):
        find_variable(data, ["temp", "lst"])


def test_find_x_y_coords():
    data = xr.Dataset(coords={"x": [0, 1], "y": [0, 1]})

    assert find_x_y_coords(data) == ("x", "y")


def test_find_x_y_coords_not_found():
    data = xr.Dataset(coords={"a": [0, 1], "b": [0, 1]})

    with pytest.raises(ValueError, match="No suitable x/y coordinate pair"):
        find_x_y_coords(data)


def test_interpolate_to_new_coords_no_crs():
    # Points in a 3x3 grid
    original_data = xr.DataArray(
        data=np.arange(18).reshape(2, 3, 3),
        coords={
            "time": np.array(["2020-01-01", "2021-01-01"], dtype="datetime64"),
            "lon": [0, 1, 2],
            "lat": [0, 1, 2],
        },
        dims=("time", "lon", "lat"),
        name="temperature",
    )
    # Asking for coordinates in the middle of each grid cell
    new_xs = xr.DataArray([0.5, 1.5], name="x")
    new_ys = xr.DataArray([0.5, 1.5], name="y")

    interpolated = interpolate_to_new_coords(
        data=original_data,
        new_xs=new_xs,
        new_ys=new_ys,
    )

    assert interpolated.dims == ("time", "y", "x")
    assert interpolated.shape == (2, 2, 2)
    # Should result in the average value of the 4 surrounding grid cell corners
    assert np.allclose(
        interpolated.values,
        np.array(
            [
                [[2, 5], [3, 6]],
                [[11, 14], [12, 15]],
            ]
        ),
    )


def test_interpolate_to_new_coords_crs():
    # Points in a 3x3 grid
    original_data = xr.DataArray(
        data=np.arange(8).reshape(2, 2, 2),
        coords={
            "time": [0, 1],
            "lon": [-48, -47],
            "lat": [77, 78],
        },
        dims=("time", "lon", "lat"),
        name="temperature",
    )
    # Points near the middle of the grid, in EPSG:3413
    new_xs = xr.DataArray([-59290.56, -48359.97], name="x")
    new_ys = xr.DataArray([-1357977.17, -1340138.99], name="y")

    interpolated = interpolate_to_new_coords(
        data=original_data,
        new_xs=new_xs,
        new_ys=new_ys,
        input_crs="EPSG:4326",
        output_crs="EPSG:3413",
    )

    assert interpolated.dims == ("time", "y", "x")
    assert interpolated.shape == (2, 2, 2)
    # Regression test
    assert np.allclose(
        interpolated.values,
        np.array(
            [
                [
                    [1.46103209, 2.42172635],
                    [1.56569501, 2.52968286],
                ],
                [
                    [5.46103209, 6.42172635],
                    [5.56569501, 6.52968286],
                ],
            ]
        ),
    )


def test_interpolate_to_new_coords_outside_range():
    # Points in a 3x3 grid
    original_data = xr.DataArray(
        data=np.arange(8).reshape(2, 2, 2),
        coords={
            "time": [0, 1],
            "lon": [-48, -47],
            "lat": [77, 78],
        },
        dims=("time", "lon", "lat"),
        name="temperature",
    )
    # Point outside the grid, in EPSG:3413
    new_xs = xr.DataArray([-26595.17], name="x")
    new_ys = xr.DataArray([-1523636.32], name="y")

    interpolated = interpolate_to_new_coords(
        data=original_data,
        new_xs=new_xs,
        new_ys=new_ys,
        input_crs="EPSG:4326",
        output_crs="EPSG:3413",
    )

    assert interpolated.dims == ("time", "y", "x")
    assert interpolated.shape == (2, 1, 1)
    # Output should be NaN due to being outside grid
    assert np.all(np.isnan(interpolated.values))


def test_replace_feature_data():
    base_data = xr.Dataset(
        {
            "temperature": (("time", "y", "x"), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            "other_var": (("time", "y", "x"), [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
        },
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1]},
    )

    new_temperature_data = xr.Dataset(
        {
            "temperature": (("time", "y", "x"), [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),
        },
        coords={"time": [1, 2], "y": [1, 2], "x": [1, 2]},
    )

    interpolated_temperature_data = xr.DataArray(
        data=np.array([[[15, 25], [35, 45]], [[55, 65], [75, 85]]]),
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1]},
        name="temperature",
    )

    with patch.object(
        mass_balance_enhance_shared,
        "interpolate_to_new_coords",
        return_value=interpolated_temperature_data,
    ):
        result = replace_feature_data(base_data, new_temperature_data, ["temp"])

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"temperature", "other_var"}
    assert np.array_equal(result["temperature"], interpolated_temperature_data)
    assert np.array_equal(result["other_var"], base_data["other_var"])
