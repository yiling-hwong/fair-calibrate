# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import warnings

import numpy as np
from fair import FAIR
from fair.interface import fill, initialise


def run_fair(cfg):
    scenarios = ["abrupt-4xCO2"]
    batch_start = cfg["batch_start"]
    batch_end = cfg["batch_end"]
    batch_size = batch_end - batch_start

    # "4xCO2" is a prescribed-forcing specie. type="volcanic" is FaIR's
    # internal category for "input_mode=forcing, no other processing" - it
    # has nothing to do with volcanoes here, it's just the category that lets
    # us hand FaIR a forcing time series directly. f.allocate() defaults
    # (forcing_efficacy=1, forcing_scale=1, tropospheric_adjustment=0,
    # forcing_temperature_feedback=0) already give a clean pass-through, so
    # fill_species_configs() is not needed for this specie.
    species = ["4xCO2"]
    properties = {
        "4xCO2": {
            "type": "volcanic",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
    }

    f = FAIR()
    f.define_time(1850, 2000, 1)
    f.define_scenarios(scenarios)
    f.define_configs(list(range(batch_start, batch_end)))
    f.define_species(species, properties)
    f.allocate()

    # climate response
    fill(
        f.climate_configs["ocean_heat_capacity"],
        np.array([cfg["c1"], cfg["c2"], cfg["c3"]]).T,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        np.array([cfg["kappa1"], cfg["kappa2"], cfg["kappa3"]]).T,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], cfg["epsilon"])
    fill(f.climate_configs["gamma_autocorrelation"], cfg["gamma"])
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.climate_configs["forcing_4co2"], cfg["forcing_4co2"])

    # abrupt-4xCO2: forcing steps instantly to forcing_4co2 and is held
    # constant for the full 150 years; climate starts from equilibrium
    initialise(f.forcing, 0)
    fill(f.forcing, cfg["forcing_4co2"], specie="4xCO2")
    initialise(f.temperature, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    # raw years 1-150 series; Gregory regression is done as a separate step
    return (
        np.array(f.temperature[1:151, 0, :, 0]),
        np.array(f.toa_imbalance[1:151, 0, :]),
    )