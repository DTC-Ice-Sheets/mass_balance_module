"""
Unit tests for prep_data_to_csv.py.

This module tests all functions in the prep_data_to_csv module, including:
- NetCDF to CSV conversion functions
- Data processing and aggregation functions
- CSV combination functions
- Plotting functions (with mocked matplotlib)

Authors: Test Suite for Tabea Rettelbach"s code
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.prep_data_to_csv import (
    combine_gris_ais_csv,
    compute_yearly_ais_mass,
    netcdf_to_csv_ais,
    netcdf_to_csv_gris,
    plot_mean_timeseries_change_gris_ais_from_csv,
    plot_mean_timeseries_gris_ais,
    plot_random_timeseries_gris_ais,
    plot_sum_timeseries_gris_ais,
)


@pytest.fixture
def sample_gris_dataset():
    """Create a sample GrIS dataset for testing."""
    # Create sample coordinate arrays
    y_coords = np.array([1000.0, 2000.0, 3000.0])
    x_coords = np.array([500.0, 1500.0])
    years = np.array([2018, 2019, 2020])

    # Create sample lat/lon grids
    lat_grid = np.array([[70.1, 70.2], [70.3, 70.4], [70.5, 70.6]])
    lon_grid = np.array([[-45.1, -45.2], [-45.3, -45.4], [-45.5, -45.6]])

    # Create sample mass balance data (y, x, time)
    # Include some NaN values to test filtering
    vmb_data = np.array(
        [
            [[100.0, 110.0, 120.0], [200.0, 210.0, 220.0]],  # y=0
            [[np.nan, np.nan, np.nan], [300.0, 310.0, 320.0]],  # y=1 (first cell all NaN)
            [[400.0, 410.0, 420.0], [500.0, 510.0, 520.0]],  # y=2
        ]
    )

    # Create error data (15% of values)
    vmb_error = np.abs(vmb_data * 0.15)

    # Create xarray dataset
    return xr.Dataset(
        {
            "VMB": (["y", "x", "Year"], vmb_data),
            "VMBer": (["y", "x", "Year"], vmb_error),
            "lat": (["y", "x"], lat_grid),
            "lon": (["y", "x"], lon_grid),
        },
        coords={"y": y_coords, "x": x_coords, "Year": years},
    )


@pytest.fixture
def sample_ais_dataset():
    """Create a sample AIS dataset for testing."""
    y_coords = np.array([1000.0, 2000.0])
    x_coords = np.array([500.0, 1500.0, 2500.0])

    # Create irregular time steps (decimal years)
    times = np.array([2018.1, 2018.5, 2018.9, 2019.2, 2019.7, 2020.1, 2020.6])

    # Create lat/lon grids
    lat_grid = np.array([[-70.1, -70.2, -70.3], [-70.4, -70.5, -70.6]])
    lon_grid = np.array([[45.1, 45.2, 45.3], [45.4, 45.5, 45.6]])

    # Create area grid (m²)
    area_grid = np.array([[1e6, 1.1e6, 1.2e6], [1.3e6, 1.4e6, 1.5e6]])

    # Create sample mass change data (time, y, x)
    # Include some zeros and NaN values
    dm_data = np.zeros((len(times), len(y_coords), len(x_coords)))
    dm_data[0] = [[0.1, 0.2, 0.0], [0.3, np.nan, 0.4]]  # 2018.1
    dm_data[1] = [[0.15, 0.25, 0.0], [0.35, 0.0, 0.45]]  # 2018.5
    dm_data[2] = [[0.12, 0.22, 0.0], [0.32, 0.1, 0.42]]  # 2018.9
    dm_data[3] = [[0.2, 0.3, 0.0], [0.4, 0.2, 0.5]]  # 2019.2
    dm_data[4] = [[0.25, 0.35, 0.0], [0.45, 0.25, 0.55]]  # 2019.7
    dm_data[5] = [[0.3, 0.4, 0.0], [0.5, 0.3, 0.6]]  # 2020.1
    dm_data[6] = [[0.35, 0.45, 0.0], [0.55, 0.35, 0.65]]  # 2020.6

    return xr.Dataset(
        {
            "dm": (["time_dec", "y", "x"], dm_data),
            "area": (["y", "x"], area_grid),
            "lat": (["y", "x"], lat_grid),
            "lon": (["y", "x"], lon_grid),
        },
        coords={"time_dec": times, "y": y_coords, "x": x_coords},
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_gris_csv_data():
    """Create sample GrIS CSV data."""
    return pd.DataFrame(
        {
            "lat": [70.1, 70.3, 70.5],
            "lon": [-45.1, -45.3, -45.5],
            "2018_mb": [100.0, 200.0, 300.0],
            "2018_mb_error": [15.0, 30.0, 45.0],
            "2019_mb": [110.0, 210.0, 310.0],
            "2019_mb_error": [16.5, 31.5, 46.5],
            "2020_mb": [120.0, 220.0, 320.0],
            "2020_mb_error": [18.0, 33.0, 48.0],
        }
    )


@pytest.fixture
def sample_ais_csv_data():
    """Create sample AIS CSV data."""
    return pd.DataFrame(
        {
            "lat": [-70.1, -70.4],
            "lon": [45.1, 45.4],
            "2019_mb": [1000.0, 2000.0],
            "2019_mb_error": [150.0, 300.0],
            "2020_mb": [1100.0, 2100.0],
            "2020_mb_error": [165.0, 315.0],
        }
    )


def test_netcdf_to_csv_gris_basic_functionality(sample_gris_dataset, temp_dir):
    """Test basic functionality of GrIS NetCDF to CSV conversion."""
    # Create temporary NetCDF file
    netcdf_path = temp_dir / "test_gris.nc"
    sample_gris_dataset.to_netcdf(netcdf_path)

    # Define output CSV path
    csv_path = temp_dir / "test_gris.csv"

    # Run the function
    netcdf_to_csv_gris(str(netcdf_path), str(csv_path))

    # Check that CSV file was created
    assert csv_path.exists()

    # Read and verify CSV content
    df = pd.read_csv(csv_path)

    # Should have 5 rows (3*2 - 1 row with all NaN values)
    assert len(df) == 5

    # Check columns
    expected_columns = [
        "lat",
        "lon",
        "2018_mb",
        "2018_mb_error",
        "2019_mb",
        "2019_mb_error",
        "2020_mb",
        "2020_mb_error",
    ]
    assert list(df.columns) == expected_columns

    # Check that no rows have all NaN values for mass balance
    mb_cols = [col for col in df.columns if "_mb" in col and "_error" not in col]
    for _, row in df.iterrows():
        assert not pd.isna(row[mb_cols]).all()


def test_netcdf_to_csv_gris_logging(sample_gris_dataset, temp_dir, caplog):
    """Test that logging works correctly."""
    netcdf_path = temp_dir / "test_gris.nc"
    sample_gris_dataset.to_netcdf(netcdf_path)
    csv_path = temp_dir / "test_gris.csv"

    with caplog.at_level(logging.INFO):
        netcdf_to_csv_gris(str(netcdf_path), str(csv_path))

    assert f"Saved CSV to {csv_path}" in caplog.text


def test_netcdf_to_csv_gris_file_not_found(temp_dir):
    """Test error handling when NetCDF file doesn"t exist."""
    nonexistent_path = temp_dir / "nonexistent.nc"
    csv_path = temp_dir / "output.csv"

    with pytest.raises(FileNotFoundError):
        netcdf_to_csv_gris(str(nonexistent_path), str(csv_path))


def test_compute_yearly_ais_mass_basic():
    """Test basic functionality of yearly mass computation."""
    # Create test data
    dm = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],  # time 0
            [[0.15, 0.25], [0.35, 0.45]],  # time 1
            [[0.2, 0.3], [0.4, 0.5]],  # time 2
        ]
    )
    area = np.array([[1e6, 2e6], [3e6, 4e6]])
    years = np.array([2018, 2018, 2019])
    y_dim = np.array([1000.0, 2000.0])
    x_dim = np.array([500.0, 1500.0])
    unique_years = np.array([2018, 2019])

    result = compute_yearly_ais_mass(dm, area, years, y_dim, x_dim, unique_years)

    # Check shape
    assert result.shape == (2, 2, 2)  # (years, y, x)

    # Check 2018 values (average of time 0 and 1)
    expected_2018_00 = ((0.1 + 0.15) / 2) * 1e6  # y=0, x=0
    expected_2018_01 = ((0.2 + 0.25) / 2) * 2e6  # y=0, x=1
    assert np.isclose(result[0, 0, 0], expected_2018_00)
    assert np.isclose(result[0, 0, 1], expected_2018_01)

    # Check 2019 values (only time 2)
    expected_2019_00 = 0.2 * 1e6  # y=0, x=0
    expected_2019_01 = 0.3 * 2e6  # y=0, x=1
    assert np.isclose(result[1, 0, 0], expected_2019_00)
    assert np.isclose(result[1, 0, 1], expected_2019_01)


def test_compute_yearly_ais_mass_with_nan_and_zero():
    """Test handling of NaN and zero values."""
    dm = np.array(
        [
            [[0.1, np.nan], [0.0, 0.4]],  # time 0
            [[0.15, 0.25], [0.0, np.nan]],  # time 1
        ]
    )
    area = np.array([[1e6, 2e6], [3e6, 4e6]])
    years = np.array([2018, 2018])
    y_dim = np.array([1000.0, 2000.0])
    x_dim = np.array([500.0, 1500.0])
    unique_years = np.array([2018])

    result = compute_yearly_ais_mass(dm, area, years, y_dim, x_dim, unique_years)

    # Check that NaN and zero values are filtered out
    # y=0, x=0: averages 0.1 and 0.15, result is (0.1+0.15)/2 * 1e6 = 0.125 * 1e6
    assert np.isclose(result[0, 0, 0], ((0.1 + 0.15) / 2) * 1e6)

    # y=0, x=1: should be 0.25 * 2e6 (only valid value)
    assert np.isclose(result[0, 0, 1], 0.25 * 2e6)

    # y=1, x=0: all values are zero, should be NaN
    assert np.isnan(result[0, 1, 0])

    # y=1, x=1: should be 0.4 * 4e6 (only valid value)
    assert np.isclose(result[0, 1, 1], 0.4 * 4e6)


def test_netcdf_to_csv_ais_basic_functionality(sample_ais_dataset, temp_dir):
    """Test basic functionality of AIS NetCDF to CSV conversion."""
    # Create temporary NetCDF file
    netcdf_path = temp_dir / "test_ais.nc"
    sample_ais_dataset.to_netcdf(netcdf_path)

    csv_path = temp_dir / "test_ais.csv"

    # Run the function
    netcdf_to_csv_ais(str(netcdf_path), str(csv_path))

    # Check that CSV file was created
    assert csv_path.exists()

    # Read and verify CSV content
    df = pd.read_csv(csv_path)

    # Should have some rows (grid cells with valid changes)
    assert len(df) > 0

    # Check that we have the expected columns for year-on-year changes
    # We expect 2019 and 2020 (differences from 2018-2019 and 2019-2020)
    expected_year_cols = ["2019_mb", "2019_mb_error", "2020_mb", "2020_mb_error"]
    for col in expected_year_cols:
        assert col in df.columns

    # Check that lat/lon columns exist
    assert "lat" in df.columns
    assert "lon" in df.columns


@patch("scripts.prep_data_to_csv.logger")
def test_netcdf_to_csv_ais_logging(mock_logger, sample_ais_dataset, temp_dir):
    """Test that logging works correctly."""
    netcdf_path = temp_dir / "test_ais.nc"
    sample_ais_dataset.to_netcdf(netcdf_path)
    csv_path = temp_dir / "test_ais.csv"

    netcdf_to_csv_ais(str(netcdf_path), str(csv_path))

    mock_logger.info.assert_called_once_with(f"Saved CSV to {csv_path}")


def test_combine_gris_ais_csv_basic(sample_gris_csv_data, sample_ais_csv_data, temp_dir):
    """Test basic functionality of CSV combination."""
    # Save sample data to CSV files
    gris_path = temp_dir / "gris.csv"
    ais_path = temp_dir / "ais.csv"
    combined_path = temp_dir / "combined.csv"

    sample_gris_csv_data.to_csv(gris_path, index=False)
    sample_ais_csv_data.to_csv(ais_path, index=False)

    # Combine CSVs
    combine_gris_ais_csv(str(gris_path), str(ais_path), str(combined_path))

    # Check that combined file exists
    assert combined_path.exists()

    # Read and verify combined CSV
    df_combined = pd.read_csv(combined_path)

    # Should have rows from both datasets
    assert len(df_combined) == len(sample_gris_csv_data) + len(sample_ais_csv_data)

    # Check that all columns from both datasets are present
    gris_cols = set(sample_gris_csv_data.columns)
    ais_cols = set(sample_ais_csv_data.columns)
    expected_cols = gris_cols.union(ais_cols)
    assert set(df_combined.columns) == expected_cols


def test_combine_gris_ais_csv_missing_columns(temp_dir):
    """Test CSV combination when datasets have different year columns."""
    # Create datasets with different year ranges
    gris_data = pd.DataFrame(
        {
            "lat": [70.1, 70.2],
            "lon": [-45.1, -45.2],
            "2018_mb": [100.0, 200.0],
            "2018_mb_error": [15.0, 30.0],
            "2019_mb": [110.0, 210.0],
            "2019_mb_error": [16.5, 31.5],
        }
    )

    ais_data = pd.DataFrame(
        {
            "lat": [-70.1, -70.2],
            "lon": [45.1, 45.2],
            "2019_mb": [1000.0, 2000.0],
            "2019_mb_error": [150.0, 300.0],
            "2020_mb": [1100.0, 2100.0],
            "2020_mb_error": [165.0, 315.0],
        }
    )

    gris_path = temp_dir / "gris.csv"
    ais_path = temp_dir / "ais.csv"
    combined_path = temp_dir / "combined.csv"

    gris_data.to_csv(gris_path, index=False)
    ais_data.to_csv(ais_path, index=False)

    combine_gris_ais_csv(str(gris_path), str(ais_path), str(combined_path))

    df_combined = pd.read_csv(combined_path)

    # Should have all year columns from both datasets
    expected_cols = [
        "lat",
        "lon",
        "2018_mb",
        "2018_mb_error",
        "2019_mb",
        "2019_mb_error",
        "2020_mb",
        "2020_mb_error",
    ]
    assert sorted(df_combined.columns) == sorted(expected_cols)

    # Missing columns should be filled with NaN (not empty strings)
    # GrIS rows should have NaN for 2020 columns
    gris_rows = df_combined.iloc[:2]  # First 2 rows are GrIS
    assert gris_rows["2020_mb"].isna().all()
    assert gris_rows["2020_mb_error"].isna().all()

    # AIS rows should have NaN for 2018 columns
    ais_rows = df_combined.iloc[2:]  # Last 2 rows are AIS
    assert ais_rows["2018_mb"].isna().all()
    assert ais_rows["2018_mb_error"].isna().all()


@patch("scripts.prep_data_to_csv.logger")
def test_combine_gris_ais_csv_logging(mock_logger, sample_gris_csv_data, sample_ais_csv_data, temp_dir):
    """Test that logging works correctly."""
    gris_path = temp_dir / "gris.csv"
    ais_path = temp_dir / "ais.csv"
    combined_path = temp_dir / "combined.csv"

    sample_gris_csv_data.to_csv(gris_path, index=False)
    sample_ais_csv_data.to_csv(ais_path, index=False)

    combine_gris_ais_csv(str(gris_path), str(ais_path), str(combined_path))

    mock_logger.info.assert_called_once_with(f"Combined CSV saved to {combined_path}")


@patch("scripts.prep_data_to_csv.plt.show")
@patch("scripts.prep_data_to_csv.secrets.choice")
def test_plot_random_timeseries_gris_ais(mock_choice, mock_show, sample_gris_dataset, sample_ais_dataset, temp_dir):
    """Test the random timeseries plotting function."""
    # Mock the random choice to return predictable values
    mock_choice.side_effect = [[0, 0], [0, 0]]  # Return first valid cell for both datasets

    # Save datasets to temporary files
    gris_path = temp_dir / "gris.nc"
    ais_path = temp_dir / "ais.nc"
    sample_gris_dataset.to_netcdf(gris_path)
    sample_ais_dataset.to_netcdf(ais_path)

    # Should run without error
    plot_random_timeseries_gris_ais(str(gris_path), str(ais_path))

    # Check that plt.show() was called
    mock_show.assert_called_once()


@patch("scripts.prep_data_to_csv.plt.show")
def test_plot_sum_timeseries_gris_ais(mock_show, sample_gris_dataset, sample_ais_dataset, temp_dir):
    """Test the sum timeseries plotting function."""
    gris_path = temp_dir / "gris.nc"
    ais_path = temp_dir / "ais.nc"
    sample_gris_dataset.to_netcdf(gris_path)
    sample_ais_dataset.to_netcdf(ais_path)

    plot_sum_timeseries_gris_ais(str(gris_path), str(ais_path))

    mock_show.assert_called_once()


@patch("scripts.prep_data_to_csv.plt.show")
def test_plot_mean_timeseries_gris_ais(mock_show, sample_gris_dataset, sample_ais_dataset, temp_dir):
    """Test the mean timeseries plotting function."""
    gris_path = temp_dir / "gris.nc"
    ais_path = temp_dir / "ais.nc"
    sample_gris_dataset.to_netcdf(gris_path)
    sample_ais_dataset.to_netcdf(ais_path)

    plot_mean_timeseries_gris_ais(str(gris_path), str(ais_path))

    mock_show.assert_called_once()


@patch("scripts.prep_data_to_csv.plt.show")
def test_plot_mean_timeseries_change_gris_ais_from_csv(mock_show, sample_gris_csv_data, sample_ais_csv_data, temp_dir):
    """Test the CSV-based plotting function."""
    gris_path = temp_dir / "gris.csv"
    ais_path = temp_dir / "ais.csv"
    sample_gris_csv_data.to_csv(gris_path, index=False)
    sample_ais_csv_data.to_csv(ais_path, index=False)

    plot_mean_timeseries_change_gris_ais_from_csv(str(gris_path), str(ais_path))

    mock_show.assert_called_once()


def test_netcdf_to_csv_gris_all_nan_data(tmp_path):
    """Test GrIS function with all NaN data."""
    # Create dataset with all NaN values
    y_coords = np.array([1000.0, 2000.0])
    x_coords = np.array([500.0, 1500.0])
    years = np.array([2018, 2019])

    # All NaN data
    vmb_data = np.full((2, 2, 2), np.nan)
    vmb_error = np.full((2, 2, 2), np.nan)
    lat_grid = np.array([[70.1, 70.2], [70.3, 70.4]])
    lon_grid = np.array([[-45.1, -45.2], [-45.3, -45.4]])

    ds = xr.Dataset(
        {
            "VMB": (["y", "x", "Year"], vmb_data),
            "VMBer": (["y", "x", "Year"], vmb_error),
            "lat": (["y", "x"], lat_grid),
            "lon": (["y", "x"], lon_grid),
        },
        coords={"y": y_coords, "x": x_coords, "Year": years},
    )

    netcdf_path = tmp_path / "all_nan_gris.nc"
    csv_path = tmp_path / "all_nan_gris.csv"
    ds.to_netcdf(netcdf_path)

    netcdf_to_csv_gris(str(netcdf_path), str(csv_path))

    # Should create an empty CSV (header only)
    df = pd.read_csv(csv_path)
    assert len(df) == 0


def test_compute_yearly_ais_mass_empty_year():
    """Test yearly mass computation with no data for a year."""
    dm = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])  # 2 time steps, 1 y, 2 x
    area = np.array([[1e6, 2e6]])
    years = np.array([2018, 2019])  # Each time step in different year
    y_dim = np.array([1000.0])
    x_dim = np.array([500.0, 1500.0])
    unique_years = np.array([2018, 2019, 2020])  # 2020 has no data

    result = compute_yearly_ais_mass(dm, area, years, y_dim, x_dim, unique_years)

    # 2020 should be all NaN
    assert np.all(np.isnan(result[2, :, :]))
