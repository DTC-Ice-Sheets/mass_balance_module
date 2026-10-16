Quick Start
===========

The package provides command-line entry points for preparing input data,
predicting mass balance, and post-processing predictions. Run the examples
from an environment where ``mass_balance_module`` is installed. For a Poetry
checkout, prefix each command with ``poetry run``.

All input ``.zarr`` paths below may be replaced with paths appropriate for
your workflow. Where noted, an input GeoZarr can also be an HTTP(S) URL.

Enhance Elevation Data
----------------------

Replace elevation-change, radar-distance, and radar-mode features in a local
GeoZarr dataset with annual values:

.. code-block:: console

   python -m mass_balance_module.mass_balance_enhance_elevation_data_entrypoint \
     --input_geozarr input_features.zarr \
     --elevation_change_data_path cumulative_elevation_change.zarr \
     --output_path enhanced_elevation_features.zarr

``input_geozarr`` and ``elevation_change_data_path`` must be local Zarr
paths.

Enhance Temperature Data
------------------------

Replace the temperature variable in a GeoZarr dataset with enhanced,
annual-mean temperature data:

.. code-block:: console

   python -m mass_balance_module.mass_balance_enhance_temperature_data_entrypoint \
     --input_geozarr input_features.zarr \
     --enhanced_temperature_data_path enhanced_temperature.zarr \
     --output_path enhanced_temperature_features.zarr

Both input datasets can be local Zarr paths or HTTP(S) URLs.

Convert Thickness to Mass Balance
---------------------------------

Convert an ice-thickness dataset to mass balance. The input must provide
``x``, ``y``, and ``time`` coordinates, CF metadata for the required
variables, and CRS metadata written with rioxarray:

.. code-block:: console

   python -m mass_balance_module.mass_balance_from_thickness_entrypoint \
     input_thickness.zarr output_mass_balance.zarr

The output is a gridded ``(time, y, x)`` mass-balance dataset in the input
coordinate system.

Predict Mass Balance
--------------------

Run the elevation-based mass balance prediction:

.. code-block:: console

   python -m mass_balance_module.mass_balance_prediction_entrypoint \
     --input_geozarr input_features.zarr \
     --output_path predicted_mass_balance.zarr

``input_geozarr`` may be a local Zarr path or an HTTP(S) URL. Optional model
inputs are ``--ice_velocity_factor``,
``--land_surface_temperature_offset``, and ``--radar_elevation_factor``.
Use ``--downsample_factor <integer>`` to downsample predictions spatially
before mass balance is derived.

Post-process Predictions
------------------------

Compare control and predicted mass-balance datasets and write the results as
JSON:

.. code-block:: console

   python -m mass_balance_module.mass_balance_postprocessing_entrypoint \
     --mass_balance_control_path control_mass_balance.zarr \
     --mass_balance_predicted_path predicted_mass_balance.zarr \
     --output_json_path postprocessed_mass_balance.json

Both mass-balance inputs may be local Zarr paths or HTTP(S) URLs.

Getting Help
------------

Every entry point provides argument-level help. For example:

.. code-block:: console

   python -m mass_balance_module.mass_balance_prediction_entrypoint --help