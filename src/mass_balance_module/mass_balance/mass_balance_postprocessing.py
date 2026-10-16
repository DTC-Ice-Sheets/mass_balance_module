"""Mass balance post-processing module."""

import xarray as xr

from mass_balance_module.mass_balance.constants import DATASET_CRS


def mass_balance_postprocess(
    mass_balance_control: xr.Dataset,
    mass_balance_predicted: xr.Dataset,
) -> dict:
    """Post-process predicted mass balance results for plotting.

    Parameters
    ----------
    mass_balance_control : xr.Dataset
        The "control" mass balance dataset, for comparison.
    mass_balance_predicted : xr.Dataset
        The predicted mass balance dataset. Can optionally have a time axis in years.

    Returns
    -------
    dict
        A dictionary containing:
        - "total_mass_balance": The total mass balance from the predicted dataset (float).
        - "control_mass_balance": The total mass balance from the control dataset (float).
        - "x_min", "x_max", "x_res": The minimum, maximum, and number of points of the x coordinate (floats).
        - "y_min", "y_max", "y_res": The minimum, maximum, and number of points of the y coordinate (floats).
        - "years": The list of years if the dataset has a time dimension, else None (ints | None).
        - "crs": The coordinate reference system of the data (string).
        - "ice_velocity": The ice velocity feature values used for prediction (2D list).
        - "land_surface_temperature": The land surface temperature feature values used for prediction (2D list).
        - "radar_elevation": The radar elevation feature values used for prediction (2D list).
        - "prediction": The predicted mass balance values (2D list).
    """
    return {
        "total_mass_balance": mass_balance_predicted.mass_balance.sum().item(),
        "control_mass_balance": mass_balance_control.mass_balance.sum().item(),
        "x_min": mass_balance_predicted.x.min().item(),
        "x_max": mass_balance_predicted.x.max().item(),
        "x_res": len(mass_balance_predicted.x),
        "y_min": mass_balance_predicted.y.min().item(),
        "y_max": mass_balance_predicted.y.max().item(),
        "y_res": len(mass_balance_predicted.y),
        "years": mass_balance_predicted.time.values.tolist() if "time" in mass_balance_predicted.dims else None,
        "crs": DATASET_CRS,
        "ice_velocity": mass_balance_predicted["iv"].values.flatten().tolist(),
        "land_surface_temperature": mass_balance_predicted["LST"].values.flatten().tolist(),
        "radar_elevation": mass_balance_predicted["RAdh"].values.flatten().tolist(),
        "prediction": mass_balance_predicted.prediction.values.flatten().tolist(),
    }
