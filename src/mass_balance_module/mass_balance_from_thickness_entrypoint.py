"""
Command line entry point for converting ice thickness to mass balance.

Implemented for ISMIP6 data but designed to be generalisable to other datasets with the appropriate CF metadata.

By default outputs a gridded (time, y, x) mass balance zarr in kg/cell/yr, preserving the input coordinate system.
"""

import logging
from pathlib import Path

import cyclopts
import xarray as xr

from mass_balance_module import helpers
from mass_balance_module.config import MASS_BALANCE_COL_NAME
from mass_balance_module.helpers import ZarrType, get_dataset_crs
from mass_balance_module.mass_balance.mass_balance_from_thickness import compute_mass_balance

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    input_zarr: ZarrType,
    output_path: Path,
) -> None:
    """
    Convert ice thickness to gridded mass balance.

    Input dataset should have a CRS written with rioxarray and its variables should be annotated with
    standard_name/long_name attributes so that ice thickness and grounded fraction (if present) can be auto-detected. If
    the dataset lacks a 'cell_area' variable, cell area will be inferred from coordinate spacing if possible (regular
    1-D x/y coordinates); otherwise an error is raised.

    Parameters
    ----------
    input_zarr : ZarrType
        Input gridded zarr with x, y, and time coordinates, CF-attributed ice thickness (and optionally grounded
        fraction) variables, and CRS metadata written by rioxarray.
    output_path : Path
        Where to write the output zarr.
    """
    logger.info("Starting mass balance computation.")
    logger.info("Input Zarr: %s", input_zarr)
    logger.info("Output path: %s", output_path)
    input_ds = xr.open_zarr(input_zarr)

    mb = compute_mass_balance(input_ds)
    # convert MB output to a dataset with the appropriate CRS metadata - needed for writing to GeoZarr
    mb_ds = mb.to_dataset(name=MASS_BALANCE_COL_NAME)
    mb_ds = mb_ds.rio.write_crs(get_dataset_crs(input_ds))

    write_mass_balance_to_geozarr(mb_ds, output_path)


def write_mass_balance_to_geozarr(mb_ds: xr.Dataset, output_path: Path) -> None:
    """
    Write the computed mass balance to a GeoZarr file with appropriate metadata.

    Parameters
    ----------
    mb_ds : xr.Dataset
        The computed mass balance dataset.
    output_path : Path
        The path where the output GeoZarr file should be written.
    """
    # update the time coordinate to be in float seconds since epoch - in line with CF conventions for time encoding
    mb_ds["time"] = helpers.datetime64_to_float_seconds(mb_ds["time"])
    mb_ds["time"] = mb_ds["time"].assign_attrs(
        {
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
            "bounds": "time_bnds",
        }
    )
    # units and calendar are specified in the encoding during the to_zarr write - remove any existing attributes
    for key in ("units", "calendar"):
        mb_ds["time"].attrs.pop(key, None)

    # set up the mass balance variable attributes - in line with CF conventions
    mb_ds[MASS_BALANCE_COL_NAME] = mb_ds[MASS_BALANCE_COL_NAME].assign_attrs(
        {
            "standard_name": MASS_BALANCE_COL_NAME,
            "long_name": "Surface mass balance flux",
            "units": "kg m-2 s-1",
            "grid_mapping": "spatial_ref",
        }
    )
    # Preserve the root-level coordinate listing expected by existing GeoZarr consumers.
    mb_ds.attrs["_ARRAY_DIMENSIONS"] = ["time", "spatial_ref", "x", "y"]

    mb_ds.to_zarr(
        output_path,
        mode="w",
        zarr_format=2,
        encoding={
            "time": {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "standard",
                "dtype": "datetime64[s]",
            },
            MASS_BALANCE_COL_NAME: {
                "_FillValue": helpers.CF_FLOAT64_FILL_VALUE,
                "chunks": helpers.calculate_chunk_shape(mb_ds[MASS_BALANCE_COL_NAME].shape),
            },
        },
        consolidated=True,
    )


if __name__ == "__main__":
    app()
