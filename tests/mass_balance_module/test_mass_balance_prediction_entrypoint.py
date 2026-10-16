"""Test for the mass balance prediction entry point."""

import re
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import cyclopts
import pytest
import xarray as xr

from mass_balance_module import mass_balance_prediction_entrypoint
from mass_balance_module.mass_balance.constants import DATASET_URL
from mass_balance_module.mass_balance_prediction_entrypoint import app


@pytest.fixture
def mock_xr__open_zarr(simple_zarr_dataset: xr.Dataset) -> Generator[xr.Dataset, None, None]:
    with patch.object(xr, "open_zarr", return_value=simple_zarr_dataset) as mock_open_zarr:
        yield mock_open_zarr


@pytest.fixture
def patch_mb_prediction_entrypoint(simple_zarr_dataset: xr.Dataset) -> Generator[None, None, None]:
    with (
        patch.object(
            mass_balance_prediction_entrypoint,
            "get_xgb_model",
            return_value=Mock(name="xgb_model"),
        ),
        patch.object(
            mass_balance_prediction_entrypoint,
            "run_elevation_prediction",
            return_value=Mock(name="predictions"),
        ),
        patch.object(
            mass_balance_prediction_entrypoint,
            "downsample_dataset",
            return_value=Mock(name="downsampled_predictions"),
        ),
        patch.object(
            mass_balance_prediction_entrypoint,
            "derive_mass_balance_from_elevation_change",
            return_value=simple_zarr_dataset,
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("test_input_source", "expects_default_dataset"),
    [
        ("valid_geozarr_file", False),
        ("valid_url", False),
        ("empty_str", True),
        ("empty_path", True),
    ],
)
@pytest.mark.usefixtures("patch_mb_prediction_entrypoint")
def test_mass_balance_prediction_entrypoint(
    mock_xr__open_zarr: xr.Dataset,
    test_input_source: str,
    expects_default_dataset: bool,
    tmp_path: Path,
    request,
) -> None:
    input_source = request.getfixturevalue(test_input_source)
    zarr_path = tmp_path / "output.zarr"

    expected_openzarr_input = DATASET_URL if expects_default_dataset else str(input_source)

    with (
        patch.object(
            sys,
            "argv",
            [
                "mass_balance_module.mass_balance_prediction_entrypoint",
                "--input_geozarr",
                str(input_source),
                "--output_path",
                str(zarr_path),
                "--ice_velocity_factor",
                "0.75",
                "--land_surface_temperature_offset",
                "2.0",
                "--radar_elevation_factor",
                "1.25",
            ],
        ),
    ):
        assert not zarr_path.exists(), "Output path should not exist before running the entrypoint."
        app(result_action="return_value")
        # validate the output zarr was written with the expected dataset content
        assert zarr_path.exists(), "Output path should exist after running the entrypoint."

        # validate that the open_zarr function was called with the correct input source
        mock_xr__open_zarr.assert_called_once_with(expected_openzarr_input)


@pytest.mark.parametrize(
    "test_source",
    ["invalid_geozarr_path", "invalid_url"],
)
def test_mass_balance_prediction_entrypoint_invalid(test_source: str, request) -> None:
    error_string = test_data = request.getfixturevalue(test_source)
    with (
        patch.object(
            sys,
            "argv",
            [
                "mass_balance_module.mass_balance_prediction_entrypoint",
                "--input_geozarr",
                str(test_data),
                "--output_path",
                "anypath/output.zarr",
                "--ice_velocity_factor",
                "0.75",
                "--land_surface_temperature_offset",
                "2.0",
                "--radar_elevation_factor",
                "1.25",
            ],
        ),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)
