"""Test for the mass balance from thickness entry point."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyproj
import xarray as xr
from use_case_2 import mass_balance_from_thickness_entrypoint


def test_main_defaults(tmp_path: Path) -> None:
    """Test main with default arguments (prep_for_selrem=True, coarsen_factor=1)."""
    from use_case_2.mass_balance_from_thickness_entrypoint import app

    output_path = tmp_path / "output.zarr"

    with (
        patch.object(
            sys,
            "argv",
            [
                "use_case_2.mass_balance_from_thickness_entrypoint",
                "path/to/input.zarr",
                str(output_path),
            ],
        ),
        patch.object(xr, "open_zarr", return_value=MagicMock()) as mock_open_zarr,
        patch.object(
            mass_balance_from_thickness_entrypoint, "compute_mass_balance", return_value=MagicMock()
        ) as mock_compute_mb,
        patch.object(
            mass_balance_from_thickness_entrypoint, "prepare_gridded_ds_for_selrem", return_value=MagicMock()
        ) as mock_prepare,
        patch.object(
            mass_balance_from_thickness_entrypoint, "get_dataset_crs", return_value=pyproj.CRS.from_epsg(4326)
        ),
        patch.object(mass_balance_from_thickness_entrypoint, "write_to_geozarr_file") as mock_write_to_geozarr,
    ):
        app(result_action="return_value")

        mock_open_zarr.assert_called_once_with(Path("path/to/input.zarr"))
        mock_compute_mb.assert_called_once_with(mock_open_zarr.return_value)
        mock_prepare.assert_called_once_with(
            mock_compute_mb.return_value.to_dataset.return_value.rio.write_crs.return_value,
            coarsen_factor=1,
        )
        mock_write_to_geozarr.assert_called_once_with(
            mock_prepare.return_value, output_path
        )


def test_main_without_prep_for_selrem(tmp_path: Path) -> None:
    """Test main with prep_for_selrem=False skips selrem preparation."""
    from use_case_2.mass_balance_from_thickness_entrypoint import app

    output_path = tmp_path / "output.zarr"

    with (
        patch.object(
            sys,
            "argv",
            [
                "use_case_2.mass_balance_from_thickness_entrypoint",
                "path/to/input.zarr",
                str(output_path),
                "--no-prep-for-selrem",
            ],
        ),
        patch.object(xr, "open_zarr", return_value=MagicMock()) as mock_open_zarr,
        patch.object(
            mass_balance_from_thickness_entrypoint, "compute_mass_balance", return_value=MagicMock()
        ) as mock_compute_mb,
        patch.object(
            mass_balance_from_thickness_entrypoint, "prepare_gridded_ds_for_selrem", return_value=MagicMock()
        ) as mock_prepare,
        patch.object(
            mass_balance_from_thickness_entrypoint, "get_dataset_crs", return_value=pyproj.CRS.from_epsg(4326)
        ),
        patch.object(mass_balance_from_thickness_entrypoint, "write_to_geozarr_file") as mock_write_to_geozarr
    ):
        app(result_action="return_value")

        mock_open_zarr.assert_called_once_with(Path("path/to/input.zarr"))
        mock_compute_mb.assert_called_once_with(mock_open_zarr.return_value)
        mock_prepare.assert_not_called()
        mock_write_to_geozarr.assert_called_once_with(
            mock_compute_mb.return_value.to_dataset.return_value.rio.write_crs.return_value,
            output_path,
        )


def test_main_with_coarsen_factor(tmp_path: Path) -> None:
    """Test main passes coarsen_factor to prepare_gridded_ds_for_selrem."""
    from use_case_2.mass_balance_from_thickness_entrypoint import app

    output_path = tmp_path / "output.zarr"

    with (
        patch.object(
            sys,
            "argv",
            [
                "use_case_2.mass_balance_from_thickness_entrypoint",
                "path/to/input.zarr",
                str(output_path),
                "--coarsen-factor",
                "4",
            ],
        ),
        patch.object(xr, "open_zarr", return_value=MagicMock()),
        patch.object(
            mass_balance_from_thickness_entrypoint, "compute_mass_balance", return_value=MagicMock()
        ) as mock_compute_mb,
        patch.object(
            mass_balance_from_thickness_entrypoint, "prepare_gridded_ds_for_selrem", return_value=MagicMock()
        ) as mock_prepare,
        patch.object(
            mass_balance_from_thickness_entrypoint, "get_dataset_crs", return_value=pyproj.CRS.from_epsg(4326)
        ),
        patch.object(mass_balance_from_thickness_entrypoint, "write_to_geozarr_file") as mock_write_to_geozarr,
    ):
        app(result_action="return_value")

        mock_prepare.assert_called_once_with(
            mock_compute_mb.return_value.to_dataset.return_value.rio.write_crs.return_value,
            coarsen_factor=4,
        )
        mock_write_to_geozarr.assert_called_once_with(
            mock_prepare.return_value, output_path
        )
