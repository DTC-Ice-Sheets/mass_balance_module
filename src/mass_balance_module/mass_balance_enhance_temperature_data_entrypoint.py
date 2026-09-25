"""Command line entry point for the mass balance temperature data enhancement module."""

from pathlib import Path

import cyclopts
import xarray as xr

from mass_balance_module.helpers import ZarrType
from mass_balance_module.mass_balance.constants import DATASET_URL
from mass_balance_module.mass_balance.mass_balance_enhance_temperature_data import (
    aggregate_year_by_mean,
    replace_temperature_data,
)

app = cyclopts.App()


@app.default
def main(
    input_geozarr: ZarrType,
    enhanced_temperature_data_path: ZarrType,
    output_path: Path,
) -> None:
    """Command line entry point for the mass balance temperature data enhancement module.

    Parameters
    ----------
    input_geozarr : ZarrType
        Path to the GeoZarr dataset containing input features. Can be a local path or a web URL.
    enhanced_temperature_data_path : ZarrType
        Path to the local zarr dataset containing the enhanced temperature data. Can be a local path or a web URL.
    output_path : Path
        Where to save the output zarr dataset, which will contain the same data as `input_geozarr` but with the
        temperature variable replaced by the enhanced version.
    """
    # TODO Temporarily fallback to placeholder dataset when input geozarr is empty
    dataset = xr.open_zarr(DATASET_URL) if input_geozarr in ("", ".") else xr.open_zarr(input_geozarr)

    enhanced_temperature_data = xr.open_zarr(enhanced_temperature_data_path)

    enhanced_temperature_data = aggregate_year_by_mean(enhanced_temperature_data)

    enhanced_dataset = replace_temperature_data(
        base_data=dataset,
        new_temperature_data=enhanced_temperature_data,
    )
    enhanced_dataset.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
