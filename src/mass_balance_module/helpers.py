"""Utility functions for use across mass_balance_module modules."""

import logging
from pathlib import Path
from typing import Annotated

import pyproj
import xarray as xr
from cyclopts import Parameter

logger = logging.getLogger(__name__)

CF_FLOAT64_FILL_VALUE = -9999.0
BYTES_PER_MIB = 1024**2
DEFAULT_ZARR_CHUNK_SIZE_MB = 100.0


def _determine_spatial_coord_names(dataset: xr.Dataset) -> tuple[str, str] | None:
    """
    Determine the names of the spatial coordinates in the dataset.

    This function has been taken from the Data Download Module. It checks for the presence of common spatial coordinate
    names in the dataset and returns them as a tuple. If no valid spatial coordinates are found, it returns None.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset from which to determine the spatial coordinate names.

    Returns
    -------
    tuple[str, str] | None
        A tuple containing the names of the x and y coordinates if found, otherwise None.
    """
    coord_names = list(dataset.coords)
    valid_coord_name_pairs = [
        ("x", "y"),
        ("projection_x_coordinate", "projection_y_coordinate"),
        ("longitude", "latitude"),
    ]
    for x, y in valid_coord_name_pairs:
        if x in coord_names and y in coord_names:
            return x, y
    return None


def get_dataset_crs(dataset: xr.Dataset) -> pyproj.CRS | None:
    """
    Retrieve the CRS details from the dataset.

    This function has been taken from the Data Download Module. It first checks for CRS written with rioxarray. Next,
    it checks the dataset for a grid mapping attribute If the CRS is not found in the grid mapping, it checks the
    dataset attributes for a `_CRS` attribute containing the WKT. If the CRS is still not found, it checks one of the
    data variable's attributes for a `_CRS` attribute with a `wkt` key (to match the current format of S1 ice velocity
    data on EODS).

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset from which to retrieve the CRS.

    Returns
    -------
    pyproj.CRS | None
        The CRS of the dataset.
    """
    import rioxarray  # noqa: F401

    # check for CRS written with rioxarray first
    try:
        rio_crs = dataset.rio.crs
        if rio_crs is not None:
            return pyproj.CRS(rio_crs)
    except AttributeError:
        pass

    crs_wkt = None
    spatial_coord_names = _determine_spatial_coord_names(dataset)

    # check for grid mapping attribute in the spatial coordinates
    if (
        spatial_coord_names is not None
        and (grid_mapping_name := dataset[spatial_coord_names[0]].attrs.get("grid_mapping")) is not None
    ):
        crs_wkt = dataset[grid_mapping_name].attrs.get("crs_wkt", None)

    # check for CRS in the dataset attributes (e.g., as written by rioxarray)
    if crs_wkt is None and "_CRS" in dataset.attrs and "crs_wkt" in dataset.attrs["_CRS"]:
        crs_wkt = dataset.attrs["_CRS"]["crs_wkt"]

    # for the last try, handle current format of S1 ice velocity data on EODS
    example_data_var = next(iter(dataset.data_vars), None)
    if crs_wkt is None and example_data_var is None:
        return None
    if crs_wkt is None and (crs := dataset[example_data_var].attrs.get("_CRS")) is not None:
        crs_wkt = crs.get("wkt", None)
    return pyproj.CRS(crs_wkt) if crs_wkt else None


def path_or_web_url_validator(type_: object, path_or_url: Path | str) -> str:  # noqa: ARG001 - cyclopts compatibility
    """Validate that the provided path is either a valid local path or a web URL.

    NOTE: Path("") resolves to the current directory (".") - which is considered a valid path.

    Parameters
    ----------
    type_ : object
        The type of the argument being validated - unused, purely required for cyclopts validator compatibility.
    path_or_url : Path | str
        The candidate path or URL to validate.

    Returns
    -------
    str
        The validated path or URL as a string.

    Raises
    ------
    ValueError
        If the provided path is neither a valid local path nor a web URL.
    """
    if not Path(path_or_url).exists() and not str(path_or_url).startswith(("http://", "https://", "s3://")):
        raise ValueError(f"The provided path '{path_or_url}' is neither a valid local path nor a web URL.")
    return str(path_or_url)


def datetime64_to_float_seconds(dt64_array: xr.DataArray) -> xr.DataArray:
    """Convert datetime64 data to seconds-resolution datetime64 values."""
    logger.info("Converting type %s to seconds since epoch", dt64_array.dtype)
    return dt64_array.astype("datetime64[s]")


def calculate_chunk_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Keep full spatial slices and fit as many time steps as possible in a 100 MiB chunk."""
    if len(shape) != 3:
        raise ValueError("Expected mass-balance dimensions in (time, y, x) order")

    max_chunk_bytes = DEFAULT_ZARR_CHUNK_SIZE_MB * BYTES_PER_MIB
    spatial_bytes = shape[1] * shape[2] * 8

    if spatial_bytes > max_chunk_bytes:
        raise ValueError("One full spatial slice exceeds the maximum chunk size")

    time_chunk = min(shape[0], int(max_chunk_bytes // spatial_bytes))
    return (time_chunk, shape[1], shape[2])


# alias for use in cyclopts Parameter definitions
type ZarrType = Annotated[str, Parameter(validator=path_or_web_url_validator)]
