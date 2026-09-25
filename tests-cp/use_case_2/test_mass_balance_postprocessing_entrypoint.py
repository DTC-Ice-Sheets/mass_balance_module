"""Test mass balance postprocessing command line entry point."""

import json
import re
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, call, patch

import cyclopts
import pytest
import xarray as xr
from use_case_2 import mass_balance_postprocessing_entrypoint
from use_case_2.mass_balance_postprocessing_entrypoint import app


@pytest.fixture
def mock_xr__open_zarr() -> Generator[xr.Dataset, None, None]:
    """Patch mass balance postprocessing entrypoint tests to allow for testing of argument parsing."""
    with patch.object(
        xr, "open_zarr", side_effect=(Mock(name="control", spec=xr.Dataset), Mock(name="predict", spec=xr.Dataset))
    ) as mock_open_zarr:
        yield mock_open_zarr


@pytest.fixture
def mock_mass_balance_postprocess(mass_balance_results_dict) -> Generator[dict, None, None]:
    """Patch mass balance postprocessing entrypoint tests to allow for testing of argument parsing."""
    with patch.object(
        mass_balance_postprocessing_entrypoint,
        "mass_balance_postprocess",
        return_value=mass_balance_results_dict,
    ) as mock_postprocess:
        yield mock_postprocess


@pytest.mark.parametrize(
    "test_input_source",
    [
        "valid_geozarr_file",
        "valid_url",
    ],
)
@pytest.mark.usefixtures("mock_mass_balance_postprocess")
def test_mass_balance_postprocessing_entrypoint_valid(
    tmp_path: Path, mock_xr__open_zarr: xr.Dataset, test_input_source: str, mass_balance_results_dict: dict, request
) -> None:
    mb_control_path = mb_predict_path = request.getfixturevalue(test_input_source)
    json_path = tmp_path / "output.json"

    with (
        patch.object(
            sys,
            "argv",
            [
                "use_case_2.mass_balance_postprocessing_entrypoint",
                "--mass-balance-control-path",
                str(mb_control_path),
                "--mass-balance-predicted-path",
                str(mb_predict_path),
                "--output-json-path",
                str(json_path),
            ],
        ),
    ):
        assert not json_path.exists()
        app(result_action="return_value")

        # verify that the output JSON file was created in the specified location
        assert json_path.is_file()
        # verify that the output JSON file contains the expected data
        with Path(json_path).open("r") as f:
            output_json = json.load(f)
        assert output_json == mass_balance_results_dict

        # verify that the input data were ingested as expected
        mock_xr__open_zarr.assert_has_calls([call(str(mb_control_path)), call(str(mb_predict_path))])
        assert mock_xr__open_zarr.call_count == 2


@pytest.mark.parametrize(
    "mb_control_path, mb_predict_path, error_string",
    [
        ("invalid_geozarr_path", "valid_geozarr_file", "invalid_geozarr_path"),
        ("valid_geozarr_file", "invalid_geozarr_path", "invalid_geozarr_path"),
        ("valid_url", "invalid_url", "invalid_url"),
        ("invalid_url", "valid_url", "invalid_url"),
    ],
)
def test_mass_balance_postprocessing_entrypoint_invalid(
    mb_control_path: str, mb_predict_path: str, error_string: str, request
) -> None:
    test_control_path = request.getfixturevalue(mb_control_path)
    test_predict_path = request.getfixturevalue(mb_predict_path)
    error_string = request.getfixturevalue(error_string)
    with (
        patch.object(
            sys,
            "argv",
            [
                "use_case_2.mass_balance_postprocessing_entrypoint",
                "--mass-balance-control-path",
                str(test_control_path),
                "--mass-balance-predicted-path",
                str(test_predict_path),
                "--output-json-path",
                "not/used/in/test.json",
            ],
        ), pytest.raises(
        cyclopts.ValidationError,
        match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
    )
    ):
        app(result_action="return_value", exit_on_error=False)
