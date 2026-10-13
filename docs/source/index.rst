DTC Ice Sheets: Mass Balance Module
===================================

The **Mass Balance Module** estimates the mass balance of the Greenland Ice
Sheet (future update will include Antarctica) from Earth Observation (EO)
data. It is one of the ice-sheet process modules of the ESA Digital Twin
Component for Ice Sheets (DTC Ice Sheets, DTC-IS), developed under the ESA
Digital Twin Earth (DTE) programme.

The module turns satellite radar-altimetry surface elevation change, together
with other EO-derived features such as ice velocity and land surface
temperature, into gridded mass balance estimates in gigatonnes per year
(Gt yr\ :sup:`-1`). In DTC-IS it feeds **Use Case 2: EU Sea-Level Response
Fingerprint**. There, its output drives the Sea-Level Response EMulator
(SELREM), which converts ice-sheet mass change into global and regional
sea-level fingerprints, with a focus on European coastlines.

The module also supports *what-if* scenarios: users can perturb the input
forcings (for example scaling ice velocity or applying a temperature offset)
and see how the predicted mass balance, and in turn the sea-level response,
changes.

.. contents:: On this page
   :local:
   :depth: 1

Scientific background
---------------------

The module follows the calibrated radar-altimetry method of
Simonsen et al. (2021), which produced a Greenland Ice Sheet mass balance
record for 1992–2020. Combining several EO datasets with machine learning
makes this a *hybrid* mass balance estimate.

**Surface elevation change from radar altimetry.**
The module builds on the long record of Ku-band radar altimetry over the
Greenland Ice Sheet. This record was first developed in the ESA Climate Change
Initiative (CCI) and is now produced operationally by the Copernicus Climate
Change Service (C3S). The standard C3S product gives monthly surface elevation
change (SEC) on a 25 km grid, from ERS-1, ERS-2, ENVISAT, CryoSat-2 and
Sentinel-3A/B (Sørensen et al., 2018). For DTC-IS, DTU runs a modified version
of the C3S processing that shortens the CryoSat-2 and Sentinel-3 data window
from 3 to 2 years and refines the grid from 25 km to **5 km**. This
higher-resolution SEC record is hosted on the DTC-IS EO data store and updated
as new data become available.

**From volume change to mass change.**
Radar altimetry measures changes in surface height, that is volume change. To
convert it to mass change you need to know how firn compaction, radar
penetration and changes in acquisition mode affect the measured signal. Following
Simonsen et al. (2021), the method learns time-varying calibration fields that
convert radar-derived elevation change into ice-equivalent (mass) elevation
change. The original model was trained to reproduce the ICESat-derived mass
balance (Sørensen et al., 2011) from a set of *a priori* features. Ten
candidate features were tested in 768 combinations. The preferred model was
chosen for its agreement with independent IMBIE and GRACE mass balance, its
consistency with the static ICESat (2003–2009) calibration field, and the
glaciological plausibility of its spatial patterns in the 1990s.

**Machine-learning model.**
The method was originally built on an AdaBoost decision-tree ensemble. The
operational implementation in this package uses a **pre-trained XGBoost
(gradient-boosted decision tree) model**. The model is downloaded at run time
and its SHA-256 checksum is checked before it is loaded. It predicts the rate
of ice-equivalent elevation change (m ice eq. yr\ :sup:`-1`) for each grid
cell from:

* the observed annual radar elevation change (``RAdh``),
* radar-altimetry acquisition features from the C3S radar-altimetry product:
  the acquisition mode (``RAmode``, LRM vs. SARIn, with CryoSat-2 SARIn
  starting over Greenland in October 2010) and the distance from the nearest
  satellite observation to the grid point (``RAdist``),
* ice surface velocity (``iv``),
* land surface temperature (``LST``),
* location (grid ``X``/``Y`` in EPSG:3413, NSIDC Polar Stereographic North).

The prediction provides mass balance in Gt yr\ :sup:`-1`. Only the pre-trained
model runs inside the digital twin. All input features are retrieved from the
DTC-IS EO data store through the Data Download Module (DDM) for the time period
the user selects.

Module workflow
---------------

The package exposes five command-line entry points that are chained together
in the DTC-IS workflow. See :doc:`quick_start` for example commands and
:doc:`api` for the full API.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry point
     - Purpose
   * - ``mass_balance_enhance_elevation_data_entrypoint``
     - Converts a cumulative, irregularly sampled elevation-change cube into
       annual (calendar-year) elevation change and rescales windows that are
       not exactly one year long. The radar-altimetry features (``RAdh``,
       ``RAdist``, ``RAmode``) in the feature dataset are then replaced with
       these updated annual values.
   * - ``mass_balance_enhance_temperature_data_entrypoint``
     - Replaces the land surface temperature feature with a new temperature
       dataset (for example the C3S land surface temperature, or projections
       from the ECMWF Climate Change Adaptation Digital Twin), aggregated to
       annual means and interpolated onto the model grid.
   * - ``mass_balance_prediction_entrypoint``
     - Applies the user-defined *what-if* perturbations, runs the XGBoost
       model and derives gridded mass balance (Gt yr\ :sup:`-1`). The grid
       can optionally be downsampled first.
   * - ``mass_balance_postprocessing_entrypoint``
     - Compares a predicted mass balance run with a control run and writes
       totals, grid metadata, input features and the prediction to JSON for
       the DTC-IS dashboard.
   * - ``mass_balance_from_thickness_entrypoint``
     - Converts modelled ice thickness, for example ISMIP6 ice-sheet model
       projections, into gridded mass balance. By default it also reshapes the
       result into the point time series expected by SELREM, so modelled
       future mass change can be used alongside the EO-based estimate.

A typical observation-driven run is:

1. Retrieve the feature dataset (GeoZarr) and, optionally, updated elevation
   change and temperature data through the DDM.
2. Enhance the feature dataset with the updated elevation and/or temperature
   data.
3. Predict the mass balance, once as a control (no perturbation) and once
   with the user's *what-if* settings.
4. Post-process the results for display, and/or pass the mass balance to
   SELREM to compute the sea-level response.

Inputs and outputs
------------------

The table below follows the input/output schema of the DTC-IS system design
(DTSD) and has been updated to match the current implementation.

.. list-table::
   :header-rows: 1
   :widths: 12 25 48 15

   * - I/O
     - Name
     - Description
     - Format
   * - Input
     - Surface elevation change
     - Radar-altimetry surface elevation change (C3S-based, 5 km), with the
       related radar features ``RAdh``, ``RAmode`` and ``RAdist``, retrieved
       from the EO data store through the DDM and subset in time to the
       user's selection.
     - Zarr data cube
   * - Input
     - Ice velocity
     - SAR-derived ice surface velocity (``iv``), retrieved through the DDM.
     - Zarr data cube
   * - Input
     - Land surface temperature
     - EO-derived ice surface temperature (``LST``), retrieved through the DDM.
       It can be replaced with other temperature data (for example climate
       projections) in the temperature enhancement step.
     - Zarr data cube
   * - Input
     - Scenario parameters
     - Multipliers/offsets that perturb the forcings (see
       `What-if scenarios`_).
     - Floating-point numbers
   * - Input
     - Ice thickness (optional)
     - Modelled ice thickness and, optionally, grounded ice fraction (e.g.
       ISMIP6), identified by CF ``standard_name``/``long_name``
       (``land_ice_thickness``, ``grounded_ice_sheet_area_fraction``). Used
       only by the thickness-to-mass-balance entry point.
     - Zarr data cube
   * - Output
     - Mass balance
     - Gridded mass balance (``mass_balance``, Gt yr\ :sup:`-1`) together with
       the predicted ice-equivalent elevation change (``prediction``,
       m ice eq. yr\ :sup:`-1`) on the EPSG:3413 grid, per year.
     - Zarr data cube
   * - Output
     - Mass balance from thickness
     - Gridded mass balance (kg cell\ :sup:`-1` yr\ :sup:`-1`), or by default
       a point time series ready for SELREM.
     - GeoZarr
   * - Output
     - Dashboard summary
     - Total predicted and control mass balance, grid extent and resolution,
       years, CRS, input features and the predicted field.
     - JSON

What-if scenarios
-----------------

DTC-IS lets users explore how the ice sheet would respond to changed forcing,
not just reproduce observations. The prediction entry point applies the
following perturbations to the input features before the model runs:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Effect
   * - ``--ice_velocity_factor``
     - ``1.0``
     - Multiplies the ice velocity feature (``iv``).
   * - ``--land_surface_temperature_offset``
     - ``0.0``
     - Adds an offset to the land surface temperature feature (``LST``).
   * - ``--radar_elevation_factor``
     - ``1.0``
     - Multiplies the observed radar elevation change feature (``RAdh``).

The temperature enhancement step also lets you replace observed surface
temperature with projected temperature fields. The C3S land surface
temperature record and runs driven by the ECMWF Climate Change Adaptation
Digital Twin have been used this way to make short-term mass balance
forecasts. Running the post-processing step on a control run and a perturbed
run gives the change in total mass balance caused by the scenario. Passing
the result to SELREM shows how that change affects regional sea level.

Validation and uncertainty
--------------------------

The module has been validated for the Greenland Ice Sheet. The
radar-altimetry volume mass balance (radar-VMB) estimate was compared with
the following independent datasets:

* satellite gravimetry from GRACE/GRACE-FO (Barletta et al., 2020),
* input–output (mass budget) estimates by Mouginot et al. (2019),
  Colgan et al. (2019) and Mankoff et al. (2021), and
* the IMBIE reconciled assessment (Otosaka et al., 2023).

Over the 28-year record (1992–2020), the radar-VMB reproduces both the
long-term trend and the interannual variability of Greenland mass balance
within the uncertainties of these independent estimates. It is strongly
correlated with the GRACE mass balance in both timing and size of the
fluctuations. Compared year by year with the mass budget record of
Mankoff et al. (2021), the root-mean-square error (RMSE) is
**86 Gt yr**\ :sup:`-1`.

The largest discrepancies occur in years of unusually low mass loss (1992,
1995–1998 and 2019), when the model underestimates how much mass the ice
sheet retained. This is a known limitation of decision-tree ensembles. They
do not capture extreme or non-linear events at the edge of the training
data well, such as the cooling after the 1991 Pinatubo eruption. The model can
also misinterpret climate extremes, so its results for such years should be
interpreted with care.

The independent validation datasets do not always agree with each other
within their stated uncertainties (2014, for example). The module therefore
reports RMSE against reference datasets as a quantitative measure of
performance. Because the model is fast, many feature-perturbation experiments
(more than 1000) can be run alongside the mass balance calculation and used
to propagate errors from start to finish through the mass balance and
SELREM chain. The Antarctic Ice Sheet will be validated against an Antarctic
mass balance assessment once the Antarctic model is released.

Limitations and ongoing development
-----------------------------------

* **Spatial coverage.** The module currently covers the Greenland Ice
  Sheet only, on the EPSG:3413 grid. The method can be transferred to
  Antarctica using the Antarctic C3S elevation-change and ice-velocity
  products. This extension is ongoing and is considered incremental research
  with low risk.
* **Extreme years.** Like other decision-tree ensembles, the model cannot
  extrapolate far outside the conditions it was trained on (see
  `Validation and uncertainty`_). Results for anomalous years and for strongly
  perturbed what-if scenarios should be read with this in mind.
* **Future projections.** Future mass change from ice-sheet model
  projections (ISMIP6; Nowicki et al., 2016) can be brought into the
  same framework through the thickness-to-mass-balance entry point. R&D
  continues on how best to combine these projections with the EO-driven
  estimate, so that user-defined what-if scenarios stay within the range of
  modelled futures.
* **Towards hybrid, explainable models.** A purely data-driven model offers
  limited insight into physical processes. In the future, the module could
  possibly move towards hybrid, explainable models that combine EO data with
  physical constraints on ice and snow processes. Approaches of interest
  include symbolic regression (e.g. Qlattice; Wilstrup and Kasak, 2021) and
  physics-informed neural networks such as PINNICLE (Cheng et al., 2024).

Related DTC-IS components
-------------------------

* **Sea-Level Response EMulator (SELREM)**, in the ``sea_level_response``
  package: turns mass balance from this module (or from user-supplied or
  ISMIP6 datasets) into global and local sea-level fingerprints with
  uncertainties.
* **Data Download Module (DDM)**: retrieves the EO input features from the
  DTC-IS EO data store and other sources.
* **DTC-IS Query API and dashboards**: run the module as part of the
  Use Case 2 workflow. Users can choose a time range and what-if parameters
  in the interactive dashboard or the accompanying Jupyter Notebook, and
  export intermediate mass balance datasets.

How to cite
-----------

If you use mass balance estimates produced by this module, please cite the
underlying method:

   Simonsen, S. B., Barletta, V. R., Colgan, W. T., & Sørensen, L. S. (2021).
   Greenland Ice Sheet mass balance (1992–2020) from calibrated radar
   altimetry. *Geophysical Research Letters*, 48, e2020GL091216.
   https://doi.org/10.1029/2020GL091216


and mention that the estimates were produced with the DTC Ice Sheets Mass
Balance Module [date].

Source documents
----------------

The scientific description on this page is based on the following DTC Ice
Sheets project deliverables (approved 1 May 2026):

* **DTU-ESA-DTC-IceSheet-RnDRdv3.00**: Research and Development Report,
  Draft Version 3, §2.1.1 Ice Sheet Mass Balance Module.
* **DTU-ESA-DTC-IceSheet-DTSDdv3.00**: Design and Technical Specifications
  Document, Draft Version 3, Ice sheet mass balance module and Use Case 2
  workflow.
* **DTU-ESA-DTC-IceSheet-VRdv2.00**: Validation Report, Draft Version 2,
  §5.1 Ice Sheet Mass Balance and SELREM.
* **DTU-ESA-DTC-IceSheet-RBDdv3.00**: Requirement Baseline Document, Draft
  Version 3, Use Case 2 requirements.
* **DTU-ESA-DTC-IceSheet-PR06v1.00**: Progress Report 6.

References
----------

* Barletta, V. R., Sørensen, L. S., Simonsen, S. B., & Forsberg, R. (2020).
  Gravitational mass balance of Greenland 2003 to present v3.0. Technical
  University of Denmark. Dataset. https://doi.org/10.11583/DTU.12866579.v5
* Cheng, G., Morlighem, M., & Francis, S. (2024). Forward and inverse modeling
  of ice sheet flow using physics-informed neural networks: Application to
  Helheim Glacier, Greenland. *JGR: Machine Learning and Computation*, 1,
  e2024JH000169. https://doi.org/10.1029/2024JH000169
* Colgan, W., Mankoff, K. D., Kjeldsen, K. K., et al. (2019). Greenland ice
  sheet mass balance assessed by PROMICE (1995–2015). *GEUS Bulletin*, 43.
  https://doi.org/10.34194/GEUSB-201943-02-01
* Mankoff, K. D., Fettweis, X., Langen, P. L., et al. (2021). Greenland ice
  sheet mass balance from 1840 through next week. *Earth System Science
  Data*, 13, 5001–5025. https://doi.org/10.5194/essd-13-5001-2021
* Mouginot, J., Rignot, E., Bjørk, A. A., et al. (2019). Forty-six years of
  Greenland Ice Sheet mass balance from 1972 to 2018. *PNAS*, 116(19),
  9239–9244. https://doi.org/10.1073/pnas.1904242116
* Nowicki, S. M. J., Payne, A., Larour, E., et al. (2016). Ice Sheet Model
  Intercomparison Project (ISMIP6) contribution to CMIP6. *Geoscientific
  Model Development*, 9, 4521–4545. https://doi.org/10.5194/gmd-9-4521-2016
* Otosaka, I. N., Shepherd, A., Ivins, E. R., et al. (2023). Mass balance of
  the Greenland and Antarctic ice sheets from 1992 to 2020. *Earth System
  Science Data*, 15, 1597–1616. https://doi.org/10.5194/essd-15-1597-2023
* Simonsen, S. B., Barletta, V. R., Colgan, W. T., & Sørensen, L. S. (2021).
  Greenland Ice Sheet mass balance (1992–2020) from calibrated radar
  altimetry. *Geophysical Research Letters*, 48, e2020GL091216.
  https://doi.org/10.1029/2020GL091216
* Sørensen, L. S., Simonsen, S. B., Nielsen, K., et al. (2011). Mass balance
  of the Greenland ice sheet (2003–2008) from ICESat data – the impact of
  interpolation, sampling and firn density. *The Cryosphere*, 5, 173–186.
  https://doi.org/10.5194/tc-5-173-2011
* Sørensen, L. S., Simonsen, S. B., Forsberg, R., et al. (2018). 25 years of
  elevation changes of the Greenland Ice Sheet from ERS, Envisat, and
  CryoSat-2 radar altimetry. *Earth and Planetary Science Letters*, 495,
  234–241. https://doi.org/10.1016/j.epsl.2018.05.015
* Wilstrup, C., & Kasak, J. (2021). Symbolic regression outperforms other
  models for small data sets. arXiv:2103.15147.
  https://arxiv.org/abs/2103.15147

.. toctree::
   :maxdepth: 2

   quick_start

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   API <api.rst>
