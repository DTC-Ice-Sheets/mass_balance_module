"""Command line entry point for the mass balance post-processing module."""

import json
from pathlib import Path

import cyclopts
import xarray as xr

from mass_balance_module.helpers import ZarrType
from mass_balance_module.mass_balance.mass_balance_postprocessing import mass_balance_postprocess

app = cyclopts.App()


@app.default
def main(
    mass_balance_control_path: ZarrType,
    mass_balance_predicted_path: ZarrType,
    output_json_path: Path,
) -> None:
    """Command line entry point for the mass balance post-processing module.

    Parameters
    ----------
    mass_balance_control_path : ZarrType
        Path to the local zarr dataset containing the control mass balance predictions. Can be a local path or a web
        URL.
    mass_balance_predicted_path : ZarrType
        Path to the local zarr dataset containing the mass balance predictions. Can be a local path or a web URL.
    output_json_path : Path
        Where to save the output JSON file.
    """
    mass_balance_control = xr.open_zarr(mass_balance_control_path)
    mass_balance_predicted = xr.open_zarr(mass_balance_predicted_path)

    out = mass_balance_postprocess(
        mass_balance_control,
        mass_balance_predicted,
    )

    with Path(output_json_path).open("w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    app()
