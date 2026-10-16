"""Unit tests for mass_balance_to_thickness module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mass_balance_module.mass_balance.constants import ICE_DENSITY
from mass_balance_module.mass_balance.mass_balance_from_thickness import (
    _find_var_by_cf_name,
    compute_mass_balance,
    derive_mass_flux_from_thickness,
    infer_cell_area,
)


@pytest.fixture
def example_thickness_ds() -> xr.Dataset:
    """Minimal (time=3, y=2, x=3) dataset with 1 km grid and linear +100 m/yr thickness increase."""
    times = pd.date_range("2010-01-01", periods=3, freq="YS")
    x = np.array([0.0, 1000.0, 2000.0])  # 1 km spacing
    y = np.array([0.0, 1000.0])  # 1 km spacing
    ny, nx = len(y), len(x)
    thickness = np.stack([np.full((ny, nx), float(v)) for v in [100.0, 200.0, 300.0]])
    return xr.Dataset(
        {
            "thickness": (
                ("time", "y", "x"),
                thickness,
                {"standard_name": "land_ice_thickness"},
            )
        },
        coords={"time": times, "y": y, "x": x},
    )


@pytest.fixture
def dt_years(example_thickness_ds: xr.Dataset) -> np.ndarray:
    """Actual year lengths for the example mb dataset's time steps (days / 365.25)."""
    return (np.diff(example_thickness_ds.time.values) / np.timedelta64(1, "D") / 365.25).astype(float)


@pytest.mark.parametrize(
    ("ds", "cf_names", "expected"),
    [
        (
            xr.Dataset({"th": xr.DataArray([1], attrs={"standard_name": "land_ice_thickness"})}),
            ["land_ice_thickness"],
            "th",
        ),
        (
            xr.Dataset({"th": xr.DataArray([1], attrs={"long_name": "Land Ice Thickness"})}),
            ["land ice thickness"],
            "th",
        ),
        (
            xr.Dataset({"th": xr.DataArray([1], attrs={"standard_name": "LAND_ICE_THICKNESS"})}),
            ["land_ice_thickness"],
            "th",
        ),
        (
            xr.Dataset({"other": xr.DataArray([1], attrs={"standard_name": "something_else"})}),
            ["land_ice_thickness"],
            None,
        ),
        (
            xr.Dataset(),
            ["land_ice_thickness"],
            None,
        ),
    ],
)
def test_find_var_by_cf_name(ds, cf_names, expected):
    assert _find_var_by_cf_name(ds, cf_names) == expected


def test_derive_mass_flux_no_grounded_frac(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """Uniform +100 m/yr increase should give 100 * ICE_DENSITY kg/m^2/yr."""
    flux = derive_mass_flux_from_thickness(example_thickness_ds, "thickness", grounded_frac_var=None)

    # 2 time steps after diff
    assert flux.dims == ("time", "y", "x")
    assert flux.sizes["time"] == 2
    # 100 m change per actual dt in years, * ICE_DENSITY
    expected_rate = 100.0 / float(dt_years[0]) * ICE_DENSITY
    np.testing.assert_equal(flux.values, expected_rate)


def test_derive_mass_flux_with_grounded_frac_ones(example_thickness_ds: xr.Dataset):
    """Grounded fraction = 1 everywhere -> same result as no grounded fraction."""
    grounded = np.ones_like(example_thickness_ds["thickness"].values)
    ds = example_thickness_ds.assign(
        grounded_frac=(
            ("time", "y", "x"),
            grounded,
            {"standard_name": "grounded_ice_sheet_area_fraction"},
        )
    )
    flux_no_frac = derive_mass_flux_from_thickness(ds, "thickness", grounded_frac_var=None)
    flux_with_frac = derive_mass_flux_from_thickness(ds, "thickness", grounded_frac_var="grounded_frac")
    np.testing.assert_equal(flux_with_frac.values, flux_no_frac.values)


def test_derive_mass_flux_with_grounded_frac_half(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """Grounded fraction = 0.5 -> flux should be halved."""
    grounded = np.full_like(example_thickness_ds["thickness"].values, 0.5)
    ds = example_thickness_ds.assign(
        grounded_frac=(
            ("time", "y", "x"),
            grounded,
            {"standard_name": "grounded_ice_sheet_area_fraction"},
        )
    )
    flux = derive_mass_flux_from_thickness(ds, "thickness", grounded_frac_var="grounded_frac")
    expected_rate = 100.0 / float(dt_years[0]) * ICE_DENSITY * 0.5
    np.testing.assert_equal(flux.values, expected_rate)


def test_derive_mass_flux_missing_grounded_frac_var(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """If grounded_frac_var is specified but absent from ds, treat as fully grounded."""
    flux = derive_mass_flux_from_thickness(example_thickness_ds, "thickness", grounded_frac_var="grounded_frac")
    expected_rate = 100.0 / float(dt_years[0]) * ICE_DENSITY
    np.testing.assert_equal(flux.values, expected_rate)


def test_derive_mass_flux_non_uniform_time(example_thickness_ds: xr.Dataset):
    """Time intervals spanning multiple years should be accounted for correctly."""
    times = pd.to_datetime(["2010-01-01", "2015-01-01"])  # 5-year gap
    ny, nx = example_thickness_ds.sizes["y"], example_thickness_ds.sizes["x"]
    # thickness goes from 100 to 600 m -> 500 m channge over 5 years -> 100 m/yr
    thickness = np.stack([np.full((ny, nx), 100.0), np.full((ny, nx), 600.0)])
    ds = xr.Dataset(
        {"thickness": (("time", "y", "x"), thickness, {"standard_name": "land_ice_thickness"})},
        coords={"time": times, "y": example_thickness_ds.coords["y"], "x": example_thickness_ds.coords["x"]},
    )
    flux = derive_mass_flux_from_thickness(ds, "thickness", grounded_frac_var=None)
    interval_years = float((pd.Timestamp("2015-01-01") - pd.Timestamp("2010-01-01")).days / 365.25)
    expected = 500.0 / interval_years * ICE_DENSITY
    np.testing.assert_equal(flux.values, expected)


def test_infer_cell_area_from_explicit_variable(example_thickness_ds: xr.Dataset):
    """Explicit cell_area variable should be returned as-is."""
    cell_area = 1000.0 * 1000.0  # 1 km^2
    ny, nx = example_thickness_ds.sizes["y"], example_thickness_ds.sizes["x"]
    area_data = np.full((ny, nx), cell_area)
    ds = example_thickness_ds.assign(cell_area=(("y", "x"), area_data))
    area = infer_cell_area(ds)
    np.testing.assert_equal(area.values, cell_area)


def test_infer_cell_area_from_regular_1d_coords(example_thickness_ds: xr.Dataset):
    """Regular 1-D x/y coords should yield dx*dy per cell."""
    area = infer_cell_area(example_thickness_ds)
    np.testing.assert_equal(float(area.to_numpy().flat[0]), 1000.0 * 1000.0)


def test_infer_cell_area_raises_no_coords():
    """Dataset without cell_area or x/y coords should raise ValueError."""
    ds = xr.Dataset({"data": (("time",), [1, 2, 3])})
    with pytest.raises(ValueError, match="cell_area"):
        infer_cell_area(ds)


def test_infer_cell_area_raises_curvilinear():
    """2-D x/y coordinate arrays (curvilinear grid) should raise ValueError."""
    x_2d = np.array([[0.0, 1000.0], [0.0, 1000.0]])
    y_2d = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    ds = xr.Dataset(
        {"thickness": (("y", "x"), np.ones((2, 2)))},
        coords={
            "x": (("y", "x"), x_2d),
            "y": (("y", "x"), y_2d),
        },
    )
    with pytest.raises(ValueError, match="Curvilinear"):
        infer_cell_area(ds)


def test_compute_mass_balance_output(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """Output should have (time-1, y, x) shape."""
    mb = compute_mass_balance(example_thickness_ds)
    assert mb.dims == ("time", "y", "x")
    assert mb.sizes["time"] == example_thickness_ds.sizes["time"] - 1
    assert mb.sizes["y"] == example_thickness_ds.sizes["y"]
    assert mb.sizes["x"] == example_thickness_ds.sizes["x"]
    expected = 100.0 / float(dt_years[0]) * ICE_DENSITY * 1000.0 * 1000.0  # 100 m/yr * ICE_DENSITY * 1 km^2
    np.testing.assert_equal(mb.values, expected)


def test_compute_mass_balance_with_explicit_cell_area(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """Explicit cell_area variable should be used in preference to inferred area."""
    ny, nx = example_thickness_ds.sizes["y"], example_thickness_ds.sizes["x"]
    area_data = np.full((ny, nx), 500_000.0)  # 0.5 km^2
    ds = example_thickness_ds.assign(cell_area=(("y", "x"), area_data))
    mb = compute_mass_balance(ds)
    expected = 100.0 / float(dt_years[0]) * ICE_DENSITY * 500_000.0

    np.testing.assert_equal(mb.values, expected)


def test_compute_mass_balance_with_grounded_frac(example_thickness_ds: xr.Dataset, dt_years: np.ndarray):
    """Grounded fraction = 0.5 should halve the output mass balance."""
    grounded = np.full_like(example_thickness_ds["thickness"].values, 0.5)
    ds = example_thickness_ds.assign(
        grounded_frac=(
            ("time", "y", "x"),
            grounded,
            {"standard_name": "grounded_ice_sheet_area_fraction"},
        )
    )
    mb = compute_mass_balance(ds)
    expected = 100.0 / float(dt_years[0]) * ICE_DENSITY * 1000 * 1000 * 0.5
    np.testing.assert_allclose(mb.values, expected, rtol=1e-5)


def test_compute_mass_balance_detects_via_long_name(example_thickness_ds: xr.Dataset):
    """Auto-detection should work with long_name as well as standard_name."""
    thickness = example_thickness_ds["thickness"].copy()
    thickness.attrs = {"long_name": "land_ice_thickness"}
    ds = example_thickness_ds.assign(thickness=thickness)
    mb = compute_mass_balance(ds)
    assert mb.sizes["time"] == example_thickness_ds.sizes["time"] - 1


def test_compute_mass_balance_raises_no_thickness_var(example_thickness_ds: xr.Dataset):
    """Missing thickness variable should raise a descriptive ValueError."""
    ds = xr.Dataset(
        {
            "unrelated": (
                ("time", "y", "x"),
                np.ones_like(example_thickness_ds["thickness"].values),
                {"standard_name": "something_else"},
            )
        },
        coords=dict(example_thickness_ds.coords),
    )
    with pytest.raises(ValueError, match="ice thickness"):
        compute_mass_balance(ds)
