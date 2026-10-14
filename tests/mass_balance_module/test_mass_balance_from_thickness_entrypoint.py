"""Test for the mass balance from thickness entry point."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyproj
import rioxarray  # noqa: F401
import xarray as xr
import zarr

from mass_balance_module import mass_balance_from_thickness_entrypoint


def test_main_defaults(tmp_path: Path, gridded_mass_balance_data_array: xr.DataArray) -> None:
    """Test the command writes gridded mass balance by default."""
    from mass_balance_module.mass_balance_from_thickness_entrypoint import app

    output_path = tmp_path / "output.zarr"
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()

    with (
        patch.object(
            sys,
            "argv",
            [
                "mass_balance_module.mass_balance_from_thickness_entrypoint",
                str(input_path),
                str(output_path),
            ],
        ),
        patch.object(xr, "open_zarr", return_value=MagicMock()) as mock_open_zarr,
        patch.object(
            mass_balance_from_thickness_entrypoint,
            "compute_mass_balance",
            return_value=gridded_mass_balance_data_array,
        ) as mock_compute_mb,
        patch.object(
            mass_balance_from_thickness_entrypoint, "get_dataset_crs", return_value=pyproj.CRS.from_epsg(4326)
        ),
    ):
        app(result_action="return_value")

        mock_open_zarr.assert_called_once_with(str(input_path))
        mock_compute_mb.assert_called_once_with(mock_open_zarr.return_value)


def test_main_writes_time_as_datetime64_seconds(tmp_path: Path, gridded_mass_balance_data_array: xr.DataArray) -> None:
    output_path = tmp_path / "output.zarr"

    with (
        patch.object(xr, "open_zarr", return_value=xr.Dataset()),
        patch.object(
            mass_balance_from_thickness_entrypoint,
            "compute_mass_balance",
            return_value=gridded_mass_balance_data_array,
        ),
        patch.object(
            mass_balance_from_thickness_entrypoint,
            "get_dataset_crs",
            return_value=pyproj.CRS.from_epsg(4326),
        ),
    ):
        mass_balance_from_thickness_entrypoint.main("input.zarr", output_path)

    output_group = zarr.open_group(output_path, mode="r")
    time_array = output_group["time"]
    assert time_array.dtype == np.dtype("datetime64[s]")


def test_write_mass_balance_to_geozarr_writes_cf_data_and_crs(
    tmp_path: Path, gridded_mass_balance_dataset: xr.Dataset
) -> None:
    output_path = tmp_path / "output.zarr"
    variable_name = mass_balance_from_thickness_entrypoint.MASS_BALANCE_COL_NAME
    mass_balance_from_thickness_entrypoint.write_mass_balance_to_geozarr(gridded_mass_balance_dataset, output_path)
    written_dataset = xr.open_zarr(output_path, decode_coords="all")
    assert written_dataset[variable_name].attrs["standard_name"] == variable_name
    assert written_dataset[variable_name].attrs["long_name"] == "Surface mass balance flux"
    assert written_dataset[variable_name].attrs["units"] == "kg m-2 s-1"
    np.testing.assert_array_equal(
        written_dataset[variable_name].values,
        gridded_mass_balance_dataset[variable_name].values,
    )
    np.testing.assert_array_equal(
        written_dataset["time"].values,
        gridded_mass_balance_dataset[variable_name]["time"].values.astype("datetime64[s]"),
    )
    assert written_dataset["time"].dtype == np.dtype("datetime64[s]")
    assert written_dataset.rio.crs == pyproj.CRS.from_epsg(4326)

    output_group = zarr.open_group(output_path, mode="r")
    expected_fill_value = mass_balance_from_thickness_entrypoint.helpers.CF_FLOAT64_FILL_VALUE
    assert output_group[variable_name].fill_value == expected_fill_value
