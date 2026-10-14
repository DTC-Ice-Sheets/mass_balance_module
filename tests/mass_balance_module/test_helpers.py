"""Tests for helper utilities."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from mass_balance_module.helpers import (
    _determine_spatial_coord_names,
    get_dataset_crs,
    path_or_web_url_validator,
    prepare_time_data_array,
)


def test_get_dataset_crs_returns_none_when_absent():
    ds = xr.Dataset({"data": (("x",), [1, 2, 3])})
    assert get_dataset_crs(ds) is None


def test_get_dataset_crs_in_ds_attributes():
    ds = xr.Dataset()
    ds.attrs["_CRS"] = {"crs_wkt": "EPSG:4326"}
    crs = get_dataset_crs(ds)
    assert crs is not None
    assert crs.to_epsg() == 4326


def test_get_dataset_crs_written_with_rioxarray():
    ds = xr.Dataset()
    ds = ds.rio.write_crs("EPSG:4326")
    crs = get_dataset_crs(ds)
    assert crs is not None
    assert crs.to_epsg() == 4326


@pytest.mark.parametrize(
    ("coords", "expected"),
    [
        ({"x": [1, 2], "y": [3, 4]}, ("x", "y")),
        (
            {"projection_x_coordinate": [1, 2], "projection_y_coordinate": [3, 4]},
            ("projection_x_coordinate", "projection_y_coordinate"),
        ),
        ({"longitude": [1, 2], "latitude": [3, 4]}, ("longitude", "latitude")),
        ({"time": [1, 2]}, None),
    ],
)
def test_determine_spatial_coord_names(coords, expected):
    ds = xr.Dataset(coords=coords)
    assert _determine_spatial_coord_names(ds) == expected


@pytest.mark.parametrize(
    "input_factory",
    [
        lambda tmp_path: tmp_path,
        lambda tmp_path: str(tmp_path),
        lambda _tmp_path: "",
        lambda _tmp_path: "http://example.com/data.zarr",
        lambda _tmp_path: "https://example.com/data.zarr",
        lambda _tmp_path: "s3://bucket/path/data.zarr",
    ],
)
def test_path_or_web_url_validator_valid_inputs(
    tmp_path: Path,
    input_factory: Callable[[Path], Path | str],
) -> None:
    candidate = input_factory(tmp_path)

    result = path_or_web_url_validator(type_=object, path_or_url=candidate)

    assert result == str(candidate)


@pytest.mark.parametrize(
    "input_factory",
    [
        lambda tmp_path: tmp_path / "missing-directory",
        lambda tmp_path: str(tmp_path / "missing-file.zarr"),
        lambda _tmp_path: "ftp://example.com/data.zarr",
    ],
)
def test_path_or_web_url_validator_invalid_inputs(
    tmp_path: Path,
    input_factory: Callable[[Path], Path | str],
) -> None:
    candidate = input_factory(tmp_path)

    with pytest.raises(ValueError, match="neither a valid local path nor a web URL"):
        path_or_web_url_validator(type_=object, path_or_url=candidate)


def test_prepare_time_data_array_converts_to_seconds_and_sets_cf_metadata(
    gridded_mass_balance_data_array: xr.DataArray,
) -> None:
    time = gridded_mass_balance_data_array["time"]
    result = prepare_time_data_array(time)
    assert result.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(result.values, time.values.astype("datetime64[s]"))
    assert result.attrs["standard_name"] == "time"
    assert result.attrs["units"] == "seconds since 1970-01-01 00:00:00"
    assert result.attrs["calendar"] == "standard"
