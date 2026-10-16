"""Command line entry point for the mass balance elevation data enhancement module."""

from pathlib import Path

import cyclopts
import xarray as xr

from mass_balance_module.mass_balance.constants import DATASET_URL
from mass_balance_module.mass_balance.mass_balance_enhance_elevation_data import (
    aggregate_year_by_annual_change,
    replace_elevation_data,
)

app = cyclopts.App()


@app.default
def main(
    input_geozarr: Path,
    elevation_change_data_path: Path,
    output_path: Path,
) -> None:
    """Command line entry point for the mass balance elevation data enhancement module.

    Parameters
    ----------
    input_geozarr : Path
        Path to the local GeoZarr dataset containing input features.
    elevation_change_data_path : Path
        Path to the local zarr dataset containing the cumulative elevation-change data.
    output_path : Path
        Where to save the output zarr dataset, which will contain the same data as `input_geozarr` but with the
        elevation-change, radar-distance and radar-mode features replaced by their per-year values.
    """
    # TODO Temporarily fallback to placeholder dataset when input geozarr is empty
    dataset = xr.open_zarr(DATASET_URL) if input_geozarr == Path() else xr.open_zarr(input_geozarr)
    elevation_change_data = xr.open_zarr(elevation_change_data_path)

    annual_elevation_change = aggregate_year_by_annual_change(elevation_change_data)

    enhanced_dataset = replace_elevation_data(
        base_data=dataset,
        new_elevation_data=annual_elevation_change,
    )
    enhanced_dataset.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
