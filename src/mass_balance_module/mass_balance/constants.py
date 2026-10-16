"""Constants for the mass balance module."""

from datetime import datetime

REMOTE_FILES_BASE_URL = (
    "http://dtc-query-api-bucket.s3-website.gra.io.cloud.ovh.net/sea_level_response/static_dash_files/"
)
DATASET_URL = REMOTE_FILES_BASE_URL + "ISMB_module_output_greenland.zarr/"
XGBOOST_MODEL_URL = REMOTE_FILES_BASE_URL + "xgb_model.pkl"
XGBOOST_MODEL_SHA256 = "75482989f76bc1802d74d7eb8d5be447e4ff01fbcaceef22ad407be3dce06a7d"

DATASET_CRS = "EPSG:3413"  # The CRS of the input dataset (NSIDC Sea Ice Polar Stereographic North)
ICE_DENSITY = 917  # Density of ice in kg/m^3
KG_TO_GT = 1e-12  # Conversion factor from kilograms to gigatonnes

# CryoSat-2 switched Greenland acquisition from LRM to SARIn here, which is what the RAmode feature encodes.
SARIN_START = datetime(2010, 10, 1)
SECONDS_PER_YEAR = 365.25 * 86400.0
# Sampling gap beyond which an elevation-change window is not treated as a full year.
ANNUAL_WINDOW_TOLERANCE = 0.02

# CF standard names used to auto-detect input variables
THICKNESS_CF_NAMES: list[str] = ["land_ice_thickness"]
GROUNDED_FRAC_CF_NAMES: list[str] = ["grounded_ice_sheet_area_fraction"]
