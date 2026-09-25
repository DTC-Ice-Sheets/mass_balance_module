"""Mass balance prediction module."""

import io
from hashlib import sha256

import cloudpickle
import numpy as np
import requests
import xarray as xr
import xgboost as xgb

from mass_balance_module.mass_balance.constants import ICE_DENSITY, KG_TO_GT, XGBOOST_MODEL_SHA256, XGBOOST_MODEL_URL
from mass_balance_module.mass_balance.mass_balance_from_thickness import infer_cell_area


def get_xgb_model() -> xgb.Booster:
    """Get the pre-trained XGBoost model."""
    response = requests.get(XGBOOST_MODEL_URL, timeout=30)
    response.raise_for_status()
    model_bytes = io.BytesIO(response.content)

    # Loading data from cloudpickle is dangerous. Verify the hash to prevent running malicious code.
    model_hash = sha256(model_bytes.getbuffer()).hexdigest()
    if model_hash != XGBOOST_MODEL_SHA256:
        raise RuntimeError("XGBoost model hash does not match. Possible data corruption or tampering.")

    # Feature names are available from `booster.feature_names`, so ignoring the second return value
    booster, _feature_names = cloudpickle.load(model_bytes)

    if not isinstance(booster, xgb.Booster):
        raise RuntimeError("Loaded object is not an XGBoost Booster. Possible data corruption or tampering.")

    return booster


def _run_prediction(xgb_model: xgb.Booster, values: np.ndarray) -> np.ndarray:
    """Run the XGBoost model on an array of feature values to get an array of predictions."""
    dmatrix = xgb.DMatrix(values.T, feature_names=xgb_model.feature_names)
    predictions = xgb_model.predict(dmatrix)
    # Set to NaN any prediction values where an input feature was NaN
    mask = np.isnan(values).any(axis=0)
    predictions[mask] = np.nan
    return predictions


def xgb_model_predict(xgb_model: xgb.Booster, dataset: xr.Dataset) -> xr.Dataset:
    """Run the XGBoost model on the input dataset to get predicted mass balance.

    Parameters
    ----------
    xgb_model : xgb.Booster
        The pre-trained XGBoost model to use for predictions.
    dataset : xr.Dataset
        The input dataset containing the features required by the model.

    Returns
    -------
    xr.Dataset
        The input dataset with an added "prediction" DataArray containing the predicted mass balance values.
    """
    # Extract just the feature names from the dataset (including X and Y coords)
    dataset_features = dataset.assign(X=dataset.x, Y=dataset.y)[xgb_model.feature_names]

    if "time" in dataset.dims:
        # Run prediction separately for each timestep
        predictions = []
        for time_idx in range(dataset.time.size):
            # Reshape input data to what the model expects: 2D array of shape (num_samples, num_features)
            data = (
                dataset_features.isel(time=time_idx)
                .to_array(dim="feature")
                .transpose("y", "x", "feature")
                .stack(sample=["y", "x"])
            )
            predictions.append(_run_prediction(xgb_model, data.values))

        # Reshape prediction back to the original spatial dimensions
        prediction = xr.DataArray(
            np.array(predictions).reshape(dataset.time.size, dataset.y.size, dataset.x.size),
            dims=("time", "y", "x"),
            coords={"time": dataset.time, "y": dataset.y, "x": dataset.x},
            name="prediction",
        )
    else:
        # Reshape input data to what the model expects: 2D array of shape (num_samples, num_features)
        data = dataset_features.to_array(dim="feature").transpose("y", "x", "feature").stack(sample=["y", "x"])

        predictions = _run_prediction(xgb_model, data.values)

        # Reshape prediction back to the original spatial dimensions
        prediction = xr.DataArray(
            predictions.reshape(dataset.y.size, dataset.x.size),
            dims=("y", "x"),
            coords={"y": dataset.y, "x": dataset.x},
            name="prediction",
        )

    # Add prediction to dataset and return
    dataset["prediction"] = prediction
    return dataset


def predictions_to_mass_balance(predictions: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """Calculate mass balance in Gt/yr based on predictions output from the XGBoost model.

    Parameters
    ----------
    predictions : xr.DataArray
        The predicted rate of elevation change values from the XGBoost model, in m/yr
    area : xr.DataArray
        The area of each grid cell in m^2

    Returns
    -------
    xr.DataArray
        The mass balance values in Gt/yr
    """
    return predictions * area * ICE_DENSITY * KG_TO_GT


def run_elevation_prediction(
    xgb_model: xgb.Booster,
    dataset: xr.Dataset,
    ice_velocity_factor: float = 1.0,
    land_surface_temperature_offset: float = 0.0,
    radar_elevation_factor: float = 1.0,
) -> xr.Dataset:
    """Calculate elevation change predictions on an input dataset, after applying the specified feature adjustments.

    Parameters
    ----------
    xgb_model : xgb.Booster
        The pre-trained XGBoost model to use for predictions.
    dataset : xr.Dataset
        The input dataset containing the features required by the model.
    ice_velocity_factor : float
        Factor to apply to the ice velocity feature before prediction.
    land_surface_temperature_offset : float
        Offset to add to the land surface temperature feature before prediction.
    radar_elevation_factor : float
        Factor to apply to the radar elevation feature before prediction.

    Returns
    -------
    xr.Dataset
        The input dataset with an added `prediction` DataArray containing the predicted elevation change values.
        These are in units of `m of ice eq/yr`
    """
    dataset["iv"] *= ice_velocity_factor
    dataset["LST"] += land_surface_temperature_offset
    dataset["RAdh"] *= radar_elevation_factor

    return xgb_model_predict(xgb_model, dataset)


def derive_mass_balance_from_elevation_change(dataset: xr.Dataset) -> xr.Dataset:
    """Calculate mass balance from elevation change for the given dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The input dataset containing the elevation change predictions and grid cell areas.

    Returns
    -------
    xr.Dataset
        The input dataset with an added "mass_balance" DataArray containing the mass balance values.
    """
    dataset["mass_balance"] = predictions_to_mass_balance(dataset["prediction"], dataset["grid_cell_area"])
    return dataset


def downsample_dataset(dataset: xr.Dataset, downsample_factor: int) -> xr.Dataset:
    """Downsample the input dataset by the given factor in both spatial dimensions.

    Has no effect if `downsample_factor` is 1 or less.

    Parameters
    ----------
    dataset : xr.Dataset
        The input dataset to downsample.
    downsample_factor : int
        The factor by which to downsample in both spatial dimensions.

    Returns
    -------
    xr.Dataset
        The downsampled dataset.
    """
    if downsample_factor > 1:
        # The cell area is invalidated by downsampling and must be re-calculated
        dataset = dataset.drop_vars("grid_cell_area")
        dataset = dataset.isel(
            y=slice(None, None, downsample_factor),
            x=slice(None, None, downsample_factor),
        )
        dataset["grid_cell_area"] = infer_cell_area(dataset)
    return dataset
