"""Test for the mass balance from thickness entry point."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.array
import numpy as np
import pyproj
import pytest
import rioxarray  # ruff: ignore[unused-import]
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


@pytest.mark.parametrize(
    ("target_chunk_size", "max_chunk_bytes", "expected_chunks", "expected_nchunks"),
    [
        ("1MB", 10**6, (2, 200, 200), 128),
        ("10MB", 10 * 10**6, (4, 500, 500), 9),
        ("50MB", 50 * 10**6, (4, 1200, 1200), 4),
    ],
)
def test_write_mass_balance_to_geozarr_respects_target_chunk_size(
    tmp_path: Path,
    target_chunk_size: str,
    max_chunk_bytes: int,
    expected_chunks: tuple[int, int, int],
    expected_nchunks: int,
) -> None:
    output_path = tmp_path / "output.zarr"
    variable_name = mass_balance_from_thickness_entrypoint.MASS_BALANCE_COL_NAME
    n_times, n_y, n_x = 4, 1500, 1500

    # 72MB float64 array, larger than every target chunk size (dask parses "MB" with decimal units)
    mass_balance = dask.array.ones((n_times, n_y, n_x), dtype=np.float64, chunks=(1, 100, 100))
    assert mass_balance.nbytes > max_chunk_bytes
    mb_ds = xr.Dataset(
        {variable_name: (("time", "y", "x"), mass_balance)},
        coords={
            "time": np.arange(n_times).astype("datetime64[Y]").astype("datetime64[s]"),
            "y": np.arange(n_y, dtype=np.float64),
            "x": np.arange(n_x, dtype=np.float64),
        },
    ).rio.write_crs("EPSG:4326")

    mass_balance_from_thickness_entrypoint.write_mass_balance_to_geozarr(
        mb_ds, output_path, target_chunk_size=target_chunk_size
    )

    written_array = zarr.open_group(output_path, mode="r")[variable_name]
    assert written_array.shape == (n_times, n_y, n_x)
    assert written_array.chunks == expected_chunks
    assert written_array.nchunks == expected_nchunks
    assert written_array.nchunks_initialized == expected_nchunks
    assert np.prod(written_array.chunks) * written_array.dtype.itemsize <= max_chunk_bytes
