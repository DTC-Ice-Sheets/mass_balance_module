"""
Command line entry point for converting ice thickness to mass balance.

Implemented for ISMIP6 data but designed to be generalisable to other datasets with the appropriate CF metadata.

By default outputs a gridded (time, y, x) mass balance zarr in kg/cell/yr, preserving the input coordinate system.
Pass --prep-for-selrem to also flatten the result to the (time, point) schema expected by the SELREM entrypoint.
"""

import logging
from pathlib import Path

import cyclopts
import xarray as xr
from sea_level_response.selrem.selrem_input_prep import get_dataset_crs, prepare_gridded_ds_for_selrem

from mass_balance_module.config import MASS_BALANCE_COL_NAME
from mass_balance_module.helpers import write_to_geozarr_file
from mass_balance_module.mass_balance.mass_balance_from_thickness import compute_mass_balance

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    input_zarr: Path,
    output_path: Path,
    prep_for_selrem: bool = True,
    coarsen_factor: int = 1,
) -> None:
    """
    Convert ice thickness to mass balance and optionally prepare for SELREM.

    Input dataset should have a CRS written with rioxarray and its variables should be annotated with
    standard_name/long_name attributes so that ice thickness and grounded fraction (if present) can be auto-detected. If
    the dataset lacks a 'cell_area' variable, cell area will be inferred from coordinate spacing if possible (regular
    1-D x/y coordinates); otherwise an error is raised.

    Parameters
    ----------
    input_zarr : Path
        Input gridded zarr with x, y, and time coordinates, CF-attributed ice thickness (and optionally grounded
        fraction) variables, and CRS metadata written by rioxarray.
    output_path : Path
        Where to write the output zarr.
    prep_for_selrem : bool
        If True, additionally flatten the gridded output to the (time, point) schema expected by the SELREM entrypoint
        — including CRS reprojection to geographic coordinates and annual resampling if needed. Default True (
        SELREM-ready output).
    coarsen_factor : int
        Spatial coarsening factor applied when --prep-for-selrem is set. Mass is summed (mass-conserving);
        coordinates are averaged. Default 1.
    """
    if not prep_for_selrem and coarsen_factor != 1:
        logger.warning("Coarsen factor should only be set when --prep-for-selrem is True. Ignoring coarsen factor.")
    input_ds = xr.open_zarr(input_zarr)

    mb = compute_mass_balance(input_ds)
    # convert MB output to a dataset with the appropriate CRS metadata - needed for writing to GeoZarr
    mb_ds = mb.to_dataset(name=MASS_BALANCE_COL_NAME)
    crs = get_dataset_crs(input_ds)
    mb_ds = mb_ds.rio.write_crs(crs)

    result = prepare_gridded_ds_for_selrem(mb_ds, coarsen_factor=coarsen_factor) if prep_for_selrem else mb_ds

    write_to_geozarr_file(result, output_path)


if __name__ == "__main__":
    app()
