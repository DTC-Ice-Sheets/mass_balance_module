"""Command line entry point for the mass balance prediction module."""

from pathlib import Path

import cyclopts
import xarray as xr

from mass_balance_module.helpers import ZarrType
from mass_balance_module.mass_balance.constants import DATASET_URL
from mass_balance_module.mass_balance.mass_balance_prediction import (
    derive_mass_balance_from_elevation_change,
    downsample_dataset,
    get_xgb_model,
    run_elevation_prediction,
)

app = cyclopts.App()


@app.default
def main(
    input_geozarr: ZarrType,
    output_path: Path,
    ice_velocity_factor: float = 1.0,
    land_surface_temperature_offset: float = 0.0,
    radar_elevation_factor: float = 1.0,
    downsample_factor: int = 1,
) -> None:
    """Command line entry point for the mass balance prediction module.

    Parameters
    ----------
    input_geozarr : ZarrType
        String specifying the location of a GeoZarr dataset. Can be a local path or a web URL.
    output_path : Path
        Where to save the output predictions zarr.
    ice_velocity_factor : float
        Factor to apply to the ice velocity feature before prediction.
    land_surface_temperature_offset : float
        Offset to add to the land surface temperature feature before prediction.
    radar_elevation_factor : float
        Factor to apply to the radar elevation feature before prediction.
    downsample_factor : int
        If >1, downsample the output by this factor in both spatial dimensions.
    """
    # TODO Temporarily fallback to placeholder dataset when input geozarr is empty
    dataset = xr.open_zarr(DATASET_URL) if input_geozarr in ("", ".") else xr.open_zarr(input_geozarr)
    xgb_model = get_xgb_model()

    predictions = run_elevation_prediction(
        xgb_model=xgb_model,
        dataset=dataset,
        ice_velocity_factor=ice_velocity_factor,
        land_surface_temperature_offset=land_surface_temperature_offset,
        radar_elevation_factor=radar_elevation_factor,
    )
    predictions = downsample_dataset(predictions, downsample_factor=downsample_factor)

    # Important to run mass balance *after* downsampling, since cell areas are different
    predictions = derive_mass_balance_from_elevation_change(predictions)

    predictions.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
