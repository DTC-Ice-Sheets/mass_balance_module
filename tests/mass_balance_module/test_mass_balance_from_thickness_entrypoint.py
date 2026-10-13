"""Test for the mass balance from thickness entry point."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyproj
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr

from mass_balance_module import helpers, mass_balance_from_thickness_entrypoint


def test_datetime64_to_float_seconds_returns_seconds_dtype() -> None:
    """The conversion result uses datetime64 seconds since epoch."""
    values = xr.DataArray([0.0, 1.0, 2.0])

    result = helpers.datetime64_to_float_seconds(values)

    assert result.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(result.values, np.array([0, 1, 2], dtype="datetime64[s]"))


def test_calculate_chunk_shape_keeps_spatial_slices_whole() -> None:
    """Chunks use the largest time extent that fits while retaining full spatial slices."""
    chunks = helpers.calculate_chunk_shape((84, 550, 337))

    assert chunks == (70, 550, 337)
    assert np.prod(chunks) * 8 <= (helpers.DEFAULT_ZARR_CHUNK_SIZE_MB * helpers.BYTES_PER_MIB)


def test_calculate_chunk_shape_rejects_oversized_spatial_slice() -> None:
    """A full spatial slice larger than the limit is rejected instead of exceeding the cap."""
    with pytest.raises(ValueError, match="One full spatial slice exceeds"):
        helpers.calculate_chunk_shape((10, 5000, 5000))


def test_write_mass_balance_uses_bounded_dimension_chunks(
    tmp_path: Path, gridded_mass_balance_data_array: xr.DataArray
) -> None:
    """The writer stores a chunk tuple whose total size respects its element budget."""
    output_path = tmp_path / "output.zarr"
    dataset = gridded_mass_balance_data_array.to_dataset(
        name=mass_balance_from_thickness_entrypoint.MASS_BALANCE_COL_NAME
    )

    mass_balance_from_thickness_entrypoint.write_mass_balance_to_geozarr(dataset, output_path)

    output_array = zarr.open_group(output_path, mode="r")[mass_balance_from_thickness_entrypoint.MASS_BALANCE_COL_NAME]
    assert len(output_array.chunks) == 3
    assert np.prod(output_array.chunks) * output_array.dtype.itemsize <= (
        helpers.DEFAULT_ZARR_CHUNK_SIZE_MB * helpers.BYTES_PER_MIB
    )


def test_main_writes_time_as_datetime64_seconds(tmp_path: Path, gridded_mass_balance_data_array: xr.DataArray) -> None:
    """The output Zarr stores time as datetime64 seconds."""
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
