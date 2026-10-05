"""
Conversion from ice thickness and optionally grounded fraction to gridded mass balance.

Developed for ISMIP6 data but designed to be generalisable to other datasets with the appropriate CF metadata.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from mass_balance_module.mass_balance.constants import GROUNDED_FRAC_CF_NAMES, ICE_DENSITY, THICKNESS_CF_NAMES

_CELL_AREA_ERROR_HINT = "Add a 'cell_area' variable to your dataset with units m^2 and dims (y, x) to resolve this."


def _find_var_by_cf_name(ds: xr.Dataset, cf_names: list[str]) -> str | None:
    """
    Return the dataset variable name whose standard_name or long_name matches.

    Matching is case-insensitive against the provided set of CF names.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search for variables with matching CF attributes.
    cf_names : list[str]
        List of CF standard_name or long_name values to match against (case-insensitive).

    Returns
    -------
    str | None
        The name of the first variable found with a matching CF attribute, or None if no match is found.
    """
    for var_name in ds.data_vars:
        var = ds[var_name]
        for attr in ("standard_name", "long_name"):
            if var.attrs.get(attr, "").lower() in cf_names:
                return var_name
    return None


def derive_mass_flux_from_thickness(
    ds: xr.Dataset,
    thickness_var: str,
    grounded_frac_var: str | None,
) -> xr.DataArray:
    """
    Derive mass flux (kg/m^2/yr) from ice thickness and optionally grounded fraction.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the ice thickness variable and optionally grounded fraction.
    thickness_var : str
        Name of the ice thickness variable.
    grounded_frac_var : str | None
        Name of the grounded ice fraction variable. If not provided, the full ice column is treated as grounded.

    Returns
    -------
    xr.DataArray
        Mass flux in kg/m^2/yr on the same spatial grid as the input.
    """
    thickness_change = ds[thickness_var].diff("time")
    time_interval_years = ds.time.diff("time") / np.timedelta64(1, "D") / 365.25
    thickness_rate = thickness_change / time_interval_years

    if grounded_frac_var is not None and grounded_frac_var in ds:
        return thickness_rate * ds[grounded_frac_var] * ICE_DENSITY
    return thickness_rate * ICE_DENSITY


def infer_cell_area(ds: xr.Dataset) -> xr.DataArray:
    """
    Return grid-cell area in m^2, using explicit `cell_area` variable or inferred uniform spacing.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset, checked for cell_area variable.

    Returns
    -------
    xr.DataArray
        Cell area in m^2.

    Raises
    ------
    ValueError
        If cell area cannot be determined from the dataset.
    """
    if "cell_area" in ds:
        return ds["cell_area"]

    if "x" in ds and "y" in ds:
        if ds.x.ndim == 2 and ds.y.ndim == 2:
            raise ValueError(
                "Curvilinear grid detected (x/y are 2-D arrays): cell area cannot be "
                "inferred from coordinate spacing alone. " + _CELL_AREA_ERROR_HINT
            )
        if ds.x.ndim == 1 and ds.y.ndim == 1 and ds.x.size > 1 and ds.y.size > 1:
            dx = float(abs(ds.x[1] - ds.x[0]))
            dy = float(abs(ds.y[1] - ds.y[0]))
            return xr.DataArray(dx * dy, dims=("y", "x"), coords={"y": ds.y, "x": ds.x})

    raise ValueError(
        "Cannot infer cell area: dataset has no 'cell_area' variable and "
        "no regular 1-D x/y coordinates from which spacing can be derived. " + _CELL_AREA_ERROR_HINT
    )


def compute_mass_balance(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert ice thickness and optionally grounded fraction to gridded mass balance (kg/cell/yr).

    Variable names are detected automatically from standard_name / long_name CF attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with CF-attributed variables and CRS metadata written by rioxarray. Should contain at minimum
        x, y, and time coordinates and a variable with standard_name or long_name of "land_ice_thickness". If a variable
        with standard_name or long_name of "grounded_ice_sheet_area_fraction" is also present, it will be used to mask
        out floating ice when computing the mass balance.

    Returns
    -------
    xr.DataArray
        Gridded mass balance in kg/cell/yr with dims (time, y, x).

    Raises
    ------
    ValueError
        If no ice thickness variable can be found.
    """
    thickness_var = _find_var_by_cf_name(ds, THICKNESS_CF_NAMES)
    if thickness_var is None:
        raise ValueError(
            f"Cannot find an ice thickness variable. Expected a variable with "
            f"'standard_name' or 'long_name' equal to one of {sorted(THICKNESS_CF_NAMES)}."
        )

    grounded_frac_var = _find_var_by_cf_name(ds, GROUNDED_FRAC_CF_NAMES)

    mb_flux = derive_mass_flux_from_thickness(ds, thickness_var, grounded_frac_var)
    area = infer_cell_area(ds)
    return mb_flux * area
