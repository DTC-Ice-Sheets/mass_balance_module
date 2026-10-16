"""Test the mass balance prediction module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
import xgboost as xgb

from mass_balance_module.mass_balance.mass_balance_prediction import (
    derive_mass_balance_from_elevation_change,
    downsample_dataset,
    predictions_to_mass_balance,
    run_elevation_prediction,
    xgb_model_predict,
)


@pytest.fixture
def feature_names() -> list[str]:
    return [
        "X",
        "Y",
        "bmelt",
        "elev",
        "slope",
        "iv",
        "firn",
        "LST",
        "RAdist",
        "RAdh",
        "RAmode",
    ]


@pytest.fixture
def mock_xgb_model(feature_names) -> MagicMock:
    mock_model = MagicMock(spec=xgb.Booster)
    mock_model.feature_names = feature_names
    return mock_model


@pytest.fixture
def mock_dataset(feature_names) -> xr.Dataset:
    x = np.linspace(1, 2, 6)
    y = np.linspace(3, 4, 4)
    data = np.linspace(5, 13, 24).reshape(4, 6)
    area = (x[1] - x[0]) * (y[1] - y[0])
    return xr.Dataset(
        data_vars={
            **dict.fromkeys(feature_names, (("y", "x"), data)),
            "grid_cell_area": (("y", "x"), np.ones((4, 6)) * area),
        },
        coords={"x": x, "y": y},
    )


@pytest.fixture
def mock_dataset_with_time(feature_names) -> xr.Dataset:
    x = np.linspace(1, 2, 6)
    y = np.linspace(3, 4, 4)
    time = np.array([2010, 2011])
    data = np.linspace(5, 13, 48).reshape(2, 4, 6)
    return xr.Dataset(
        data_vars={
            **dict.fromkeys(feature_names, (("time", "y", "x"), data)),
            "grid_cell_area": (("time", "y", "x"), np.ones((2, 4, 6))),
        },
        coords={"x": x, "y": y, "time": time},
    )


@pytest.fixture
def mock_predictions(mock_dataset) -> xr.DataArray:
    return xr.DataArray(
        np.linspace(0, 23, 24).reshape(4, 6),
        dims=("y", "x"),
        coords={"y": mock_dataset.coords["y"], "x": mock_dataset.coords["x"]},
        name="prediction",
    )


def test_xgb_model_predict(mock_xgb_model, mock_dataset):
    def mock_predict(dmatrix):
        assert dmatrix.feature_names == mock_xgb_model.feature_names
        assert dmatrix.num_col() == len(mock_xgb_model.feature_names)
        assert dmatrix.num_row() == mock_dataset.y.size * mock_dataset.x.size
        num_samples = dmatrix.num_row()
        return np.arange(num_samples).astype(float)

    mock_xgb_model.predict.side_effect = mock_predict

    out = xgb_model_predict(mock_xgb_model, mock_dataset)

    assert isinstance(out, xr.Dataset)
    assert out["prediction"].dims == ("y", "x")
    assert out.coords["x"].equals(mock_dataset.coords["x"])
    assert out.coords["y"].equals(mock_dataset.coords["y"])
    assert "prediction" in out.data_vars


def test_predictions_to_mass_balance(mock_predictions, mock_dataset):
    mass_balance = predictions_to_mass_balance(mock_predictions, mock_dataset["grid_cell_area"])
    assert np.isclose(mass_balance.sum(), 1.68728e-8)


@pytest.mark.parametrize(
    (
        "ice_velocity_factor",
        "land_surface_temperature_offset",
        "radar_elevation_factor",
    ),
    [
        (1.0, 0.0, 1.0),
        (1.5, 0.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 0.0, 1.5),
        (0.8, 0.5, 0.7),
    ],
)
def test_run_mass_balance_prediction(
    mock_dataset,
    mock_xgb_model,
    ice_velocity_factor,
    land_surface_temperature_offset,
    radar_elevation_factor,
):
    def mock_predict(dmatrix):
        assert dmatrix.feature_names == mock_xgb_model.feature_names
        assert dmatrix.num_col() == len(mock_xgb_model.feature_names)
        assert dmatrix.num_row() == mock_dataset.y.size * mock_dataset.x.size
        data = dmatrix.get_data().toarray()
        assert np.allclose(mock_dataset["iv"].values.flatten() * ice_velocity_factor, data[:, 5])
        assert np.allclose(mock_dataset["LST"].values.flatten() + land_surface_temperature_offset, data[:, 7])
        assert np.allclose(mock_dataset["RAdh"].values.flatten() * radar_elevation_factor, data[:, 9])
        num_samples = dmatrix.num_row()
        return np.arange(num_samples).astype(float)

    mock_xgb_model.predict.side_effect = mock_predict

    predictions = run_elevation_prediction(
        xgb_model=mock_xgb_model,
        dataset=mock_dataset.copy(deep=True),
        ice_velocity_factor=ice_velocity_factor,
        land_surface_temperature_offset=land_surface_temperature_offset,
        radar_elevation_factor=radar_elevation_factor,
    )
    predictions = derive_mass_balance_from_elevation_change(predictions)
    assert mock_xgb_model.predict.call_count == 1

    assert isinstance(predictions, xr.Dataset)
    assert np.isclose(predictions.mass_balance.sum(), 1.68728e-8)
    assert predictions.prediction.dims == ("y", "x")
    assert predictions.coords["x"].equals(mock_dataset.coords["x"])
    assert predictions.coords["y"].equals(mock_dataset.coords["y"])


def test_run_mass_balance_prediction_with_time(
    mock_dataset_with_time,
    mock_xgb_model,
):
    def mock_predict(dmatrix):
        assert dmatrix.feature_names == mock_xgb_model.feature_names
        assert dmatrix.num_col() == len(mock_xgb_model.feature_names)
        assert dmatrix.num_row() == mock_dataset_with_time.y.size * mock_dataset_with_time.x.size
        num_samples = dmatrix.num_row()
        return np.arange(num_samples).astype(float)

    mock_xgb_model.predict.side_effect = mock_predict

    predictions = run_elevation_prediction(
        xgb_model=mock_xgb_model,
        dataset=mock_dataset_with_time.copy(deep=True),
        ice_velocity_factor=1.0,
        land_surface_temperature_offset=0.0,
        radar_elevation_factor=1.0,
    )
    predictions = derive_mass_balance_from_elevation_change(predictions)
    assert mock_xgb_model.predict.call_count == 2  # once per timestep

    assert isinstance(predictions, xr.Dataset)
    assert np.isclose(predictions.mass_balance.sum(), 5.06184e-07)
    assert predictions.prediction.dims == ("time", "y", "x")
    assert predictions.coords["x"].equals(mock_dataset_with_time.coords["x"])
    assert predictions.coords["y"].equals(mock_dataset_with_time.coords["y"])
    assert predictions.coords["time"].equals(mock_dataset_with_time.coords["time"])


def test_downsample_dataset(mock_dataset):
    new_dataset = downsample_dataset(mock_dataset, downsample_factor=2)
    assert not new_dataset.equals(mock_dataset)
    assert set(new_dataset.dims) == set(mock_dataset.dims)
    assert new_dataset.coords["x"].equals(mock_dataset.coords["x"][::2])
    assert new_dataset.coords["y"].equals(mock_dataset.coords["y"][::2])
    for var in mock_dataset.data_vars:
        assert var in new_dataset.data_vars
        assert new_dataset[var].dims == ("y", "x")
        assert new_dataset[var].shape == (2, 3)
        if var == "grid_cell_area":
            # Total area should be approximately preserved
            assert np.isclose(new_dataset[var].sum(), mock_dataset[var].sum())
            continue
        assert np.allclose(new_dataset[var].values, mock_dataset[var].values[::2, ::2])


def test_downsample_dataset_no_change(mock_dataset):
    new_dataset = downsample_dataset(mock_dataset, downsample_factor=1)
    assert new_dataset.equals(mock_dataset)
