"""Test for the mass balance enhance temperature data entry point."""

import re
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, call, patch

import cyclopts
import pytest
import xarray as xr

from mass_balance_module import mass_balance_enhance_temperature_data_entrypoint
from mass_balance_module.mass_balance.constants import DATASET_URL
from mass_balance_module.mass_balance_enhance_temperature_data_entrypoint import app


@pytest.fixture
def mock_xr__open_zarr() -> Generator[xr.Dataset, None, None]:
    with patch.object(
        mass_balance_enhance_temperature_data_entrypoint,
        "xr",
        autospec=True,
    ) as mock_xr_module:
        mock_xr_module.open_zarr.side_effect = (
            Mock(name="control", spec=xr.Dataset),
            Mock(name="predict", spec=xr.Dataset),
        )
        yield mock_xr_module.open_zarr


@pytest.fixture
def mock_aggregate_year_by_mean(simple_zarr_dataset) -> Generator[xr.Dataset, None, None]:
    with (
        patch.object(
            mass_balance_enhance_temperature_data_entrypoint,
            "aggregate_year_by_mean",
            return_value=simple_zarr_dataset,
        ),
        patch.object(
            mass_balance_enhance_temperature_data_entrypoint,
            "replace_temperature_data",
            return_value=simple_zarr_dataset,
        ),
    ):
        yield


# our test matrix should test files and urls for both input_geozarr and enhanced_temperature_data_path
# The field "expects_default_dataset" is required to denote cases where we expect DATASET_URL to be injected
@pytest.mark.parametrize(
    "input_geozarr_source, expects_default_dataset, input_temperature_data_source",
    [
        ("empty_str", True, "valid_geozarr_file"),
        ("empty_path", True, "valid_url"),
        ("valid_url", False, "empty_path"),
        ("valid_geozarr_file", False, "empty_str"),
    ],
)
@pytest.mark.usefixtures("mock_aggregate_year_by_mean")
def test_mass_balance_enhance_temperature_data_entrypoint_valid(
    input_geozarr_source: str,
    expects_default_dataset: bool,
    input_temperature_data_source: str,
    mock_xr__open_zarr: xr.Dataset,
    tmp_path: Path,
    simple_zarr_dataset: xr.Dataset,
    request,
) -> None:
    input_geozarr = str(request.getfixturevalue(input_geozarr_source))
    input_temperature_data = str(request.getfixturevalue(input_temperature_data_source))
    expected_openzarr_input = DATASET_URL if expects_default_dataset else str(input_geozarr)

    output_path_str = str(tmp_path / "output.zarr")
    with (
        patch.object(
            sys,
            "argv",
            [
                "mass_balance_module.mass_balance_enhance_temperature_data_entrypoint",
                "--input_geozarr",
                input_geozarr,
                "--enhanced_temperature_data_path",
                input_temperature_data,
                "--output_path",
                output_path_str,
            ],
        ),
    ):
        # verify that the output path does not exist before running the entrypoint
        assert not Path(output_path_str).exists()
        # run the entrypoint with the provided arguments
        app(result_action="return_value")
        # verify that the output path exists after running the entrypoint
        assert Path(output_path_str).exists()
        assert xr.open_zarr(output_path_str).equals(simple_zarr_dataset)

        # verify that the input data were ingested as expected
        mock_xr__open_zarr.assert_has_calls([call(str(expected_openzarr_input)), call(input_temperature_data)])
        assert mock_xr__open_zarr.call_count == 2


@pytest.mark.parametrize(
    "geozarr_source, enhanced_temperature_data_source",
    [
        ("invalid_url", "valid_geozarr_file"),
        ("invalid_geozarr_path", "valid_geozarr_file"),
        ("valid_geozarr_file", "invalid_url"),
        ("valid_geozarr_file", "invalid_geozarr_path"),
    ],
)
def test_invalid_path_raises_validation_error(
    geozarr_source: str, enhanced_temperature_data_source: str, request
) -> None:
    input_geozarr = str(request.getfixturevalue(geozarr_source))
    input_temperature_data_zarr = str(request.getfixturevalue(enhanced_temperature_data_source))
    invalid_fixture = geozarr_source if "invalid" in geozarr_source else enhanced_temperature_data_source
    error_string = str(request.getfixturevalue(invalid_fixture))

    with (
        patch.object(
            sys,
            "argv",
            [
                "mass_balance_module.mass_balance_enhance_temperature_data_entrypoint",
                "--input_geozarr",
                input_geozarr,
                "--enhanced_temperature_data_path",
                input_temperature_data_zarr,
                "--output_path",
                "path/to/output.zarr",
            ],
        ),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)
