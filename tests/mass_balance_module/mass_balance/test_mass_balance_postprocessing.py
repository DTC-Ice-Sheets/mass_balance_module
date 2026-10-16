"""Test for mass balance postprocessing module."""

import numpy as np
import pytest
import xarray as xr

from mass_balance_module.mass_balance.mass_balance_postprocessing import mass_balance_postprocess


@pytest.fixture
def mass_balance_control():
    return xr.Dataset(
        data_vars={
            "prediction": (("y", "x"), np.arange(9).reshape(3, 3)),
            "mass_balance": (("y", "x"), np.arange(9).reshape(3, 3)),
            "iv": (("y", "x"), np.arange(9).reshape(3, 3)),
            "LST": (("y", "x"), np.arange(9).reshape(3, 3)),
            "RAdh": (("y", "x"), np.arange(9).reshape(3, 3)),
        },
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
    )


def test_mass_balance_postprocess(mass_balance_control):
    mass_balance_predictions = xr.Dataset(
        data_vars={
            "prediction": (("y", "x"), np.arange(9).reshape(3, 3) + 5),
            "mass_balance": (("y", "x"), np.arange(9).reshape(3, 3) + 5),
            "iv": (("y", "x"), np.arange(9).reshape(3, 3) + 1),
            "LST": (("y", "x"), np.arange(9).reshape(3, 3) + 2),
            "RAdh": (("y", "x"), np.arange(9).reshape(3, 3) + 3),
        },
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
    )

    out = mass_balance_postprocess(
        mass_balance_control,
        mass_balance_predictions,
    )
    assert out.keys() == {
        "total_mass_balance",
        "control_mass_balance",
        "x_min",
        "x_max",
        "x_res",
        "y_min",
        "y_max",
        "y_res",
        "years",
        "crs",
        "ice_velocity",
        "land_surface_temperature",
        "radar_elevation",
        "prediction",
    }
    assert out["total_mass_balance"] == 81
    assert out["control_mass_balance"] == 36
    assert out["x_min"] == 0
    assert out["x_max"] == 2
    assert out["x_res"] == 3
    assert out["y_min"] == 0
    assert out["y_max"] == 2
    assert out["y_res"] == 3
    assert out["years"] is None
    assert out["crs"] == "EPSG:3413"
    assert out["ice_velocity"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert out["land_surface_temperature"] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert out["radar_elevation"] == [3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert out["prediction"] == [5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_mass_balance_postprocess_with_years(mass_balance_control):
    mass_balance_predictions = xr.Dataset(
        data_vars={
            "prediction": (("time", "y", "x"), np.arange(27).reshape(3, 3, 3) + 5),
            "mass_balance": (("time", "y", "x"), np.arange(27).reshape(3, 3, 3) + 5),
            "iv": (("time", "y", "x"), np.arange(27).reshape(3, 3, 3) + 1),
            "LST": (("time", "y", "x"), np.arange(27).reshape(3, 3, 3) + 2),
            "RAdh": (("time", "y", "x"), np.arange(27).reshape(3, 3, 3) + 3),
        },
        coords={"time": [2010, 2011, 2012], "y": [0, 1, 2], "x": [0, 1, 2]},
    )

    out = mass_balance_postprocess(
        mass_balance_control,
        mass_balance_predictions,
    )
    assert out.keys() == {
        "total_mass_balance",
        "control_mass_balance",
        "x_min",
        "x_max",
        "x_res",
        "y_min",
        "y_max",
        "y_res",
        "years",
        "crs",
        "ice_velocity",
        "land_surface_temperature",
        "radar_elevation",
        "prediction",
    }
    assert out["total_mass_balance"] == 486
    assert out["control_mass_balance"] == 36
    assert out["x_min"] == 0
    assert out["x_max"] == 2
    assert out["x_res"] == 3
    assert out["y_min"] == 0
    assert out["y_max"] == 2
    assert out["y_res"] == 3
    assert out["years"] == [2010, 2011, 2012]
    assert out["crs"] == "EPSG:3413"
    assert out["ice_velocity"] == list(range(1, 28))
    assert out["land_surface_temperature"] == list(range(2, 29))
    assert out["radar_elevation"] == list(range(3, 30))
    assert out["prediction"] == list(range(5, 32))
