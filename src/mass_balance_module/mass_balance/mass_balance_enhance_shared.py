"""Shared support for the mass balance feature-enhancement modules."""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from mass_balance_module.mass_balance.constants import DATASET_CRS


def find_variable(data: xr.Dataset, keywords: Sequence[str]) -> str:
    """Return the name of the first data variable in `data` whose name contains one of `keywords`.

    Parameters
    ----------
    data : xr.Dataset
        The dataset to search.
    keywords : Sequence[str]
        Substrings to look for in each variable name, matched case-insensitively.

    Returns
    -------
    str
        The name of the first matching variable.

    Raises
    ------
    ValueError
        If no variable name contains any of `keywords`.
    """
    for var_name in data.data_vars:
        if any(key in str(var_name).lower() for key in keywords):
            return str(var_name)
    raise ValueError(f"No variable matching {list(keywords)} found in the dataset.")


def find_x_y_coords(data: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    """Return (x_coord, y_coord) names found in `data`."""
    for target_x, target_y in [("x", "y"), ("lon", "lat"), ("longitude", "latitude")]:
        if target_x in data.dims and target_y in data.dims:
            return target_x, target_y
    raise ValueError("No suitable x/y coordinate pair found in the dataset.")


def interpolate_to_new_coords(
    data: xr.DataArray,
    new_xs: xr.DataArray,
    new_ys: xr.DataArray,
    input_crs: str | None = None,
    output_crs: str | None = None,
) -> xr.DataArray:
    """
    Linearly interpolate the input data to a new set of x/y coordinates.

    Supports optional reprojection if the input and output coordinates are in different CRSs.

    Parameters
    ----------
    data : xr.DataArray
        The data to interpolate. Must have two spatial dimensions and one other (e.g. time) dimension.
        The spatial dimensions must be named x/y, lon/lat, or longitude/latitude.
    new_xs : xr.DataArray
        The x coordinates to interpolate to.
    new_ys : xr.DataArray
        The y coordinates to interpolate to.
    input_crs : str | None
        The CRS of the input data's coordinates, if performing reprojection.
        Must be provided together with output_crs.
    output_crs : str | None
        The CRS of the output coordinates (i.e. the CRS of new_xs and new_ys), if performing reprojection.
        Must be provided together with input_crs.

    Returns
    -------
    xr.DataArray
        The new data, interpolated to the new coordinates.
        Will have the same dimensions as the input data, but with spatial dimensions replaced with new_xs and new_ys.

    Raises
    ------
    ValueError
        If no suitable x/y coordinate pair is found in the input data.
    """
    if (input_crs is None) != (output_crs is None):
        raise ValueError("Must provide either both input_crs and output_crs, or neither.")

    input_x_dim, input_y_dim = find_x_y_coords(data)
    if len(data.dims) != 3:
        raise NotImplementedError("Only 3D data (time, y, x) is currently supported for interpolation.")

    # Ensure y and x are the last two dimensions, for easier indexing later
    data = data.transpose(..., input_y_dim, input_x_dim)

    # Get meshgrid of input coordinates
    output_xs, output_ys = np.meshgrid(new_xs, new_ys, indexing="xy")

    # Get meshgrid of input coordinates, in the output CRS
    input_xs, input_ys = np.meshgrid(data[input_x_dim], data[input_y_dim], indexing="xy")
    if input_crs is not None:
        transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        input_xs, input_ys = transformer.transform(input_xs, input_ys)

    # Flatten source coordinates once; values will be flattened per timestep
    points = np.column_stack((input_xs.ravel(), input_ys.ravel()))
    points_delaunay = Delaunay(points)

    # Preallocate output array: (years, ny, nx)
    out = np.empty((data.shape[0], len(new_ys), len(new_xs)), dtype=float)

    for ii in range(data.shape[0]):
        # mean_temp[ii, :, :] has shape (nlat, nlon); flatten to match points
        values = np.squeeze(data.values[ii, :])
        values_flat = values.ravel()

        ft = LinearNDInterpolator(points_delaunay, values_flat)
        # Evaluate using (x, y) order consistent with how points were built
        intval = ft(output_xs, output_ys)
        out[ii, :, :] = intval

    # Place back into DataArray
    return xr.DataArray(
        out,
        dims=(data.dims[0], new_ys.name, new_xs.name),
        coords={
            data.dims[0]: data.coords[data.dims[0]].values,
            new_ys.name: new_ys.values,
            new_xs.name: new_xs.values,
        },
        name=data.name,
    )


def replace_feature_data(base_data: xr.Dataset, new_feature_data: xr.Dataset, keywords: Sequence[str]) -> xr.Dataset:
    """Replace one feature variable in `base_data` with a newer version of it, found in `new_feature_data`.

    The two datasets may name the feature differently; both are matched against the same `keywords`. The
    replacement data will be linearly interpolated to `base_data`'s spatial coordinates before being merged in.

    Parameters
    ----------
    base_data : xr.Dataset
        The original dataset containing the feature variable to be replaced.
    new_feature_data : xr.Dataset
        The dataset containing the improved feature data.
    keywords : Sequence[str]
        Substrings used to find the feature's variable name in both datasets, via :func:`find_variable`.

    Returns
    -------
    xr.Dataset
        Modified `base_data`, with the matched variable replaced by the interpolated `new_feature_data`.
    """
    base_var = find_variable(base_data, keywords)
    new_var = find_variable(new_feature_data, keywords)

    base_feature = base_data[base_var]
    new_feature = new_feature_data[new_var]

    x_dim, y_dim = find_x_y_coords(base_feature)

    # Remap the new feature data to the same grid as the rest of the data
    interpolated = interpolate_to_new_coords(
        data=new_feature,
        new_xs=base_feature.coords[x_dim],
        new_ys=base_feature.coords[y_dim],
        input_crs="EPSG:4326",
        output_crs=DATASET_CRS,
    )

    # Replace the original feature data with the new version
    base_data[base_var] = interpolated

    return base_data
