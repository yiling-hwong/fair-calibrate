# put imports outside: we don't have a lot of overhead here, and it looks nicer.
import warnings

import numpy as np
from fair import FAIR
from fair.interface import fill, initialise


def run_fair(cfg):
    scenarios = ["esm-flat10"]
    batch_start = cfg["batch_start"]
    batch_end = cfg["batch_end"]
    batch_size = batch_end - batch_start

    # CH4 and N2O are included in concentration mode, held constant at their
    # pre-industrial reference concentration for the whole run, so they
    # contribute zero forcing of their own (sqrt(C) - sqrt(C_ref) = 0) while
    # satisfying FaIR's requirement that for ghg_method=meinshausen2020, CO2,
    # CH4 and N2O must either all be concentration/emissions-driven, or none.
    species = ["CO2", "CH4", "N2O"]
    properties = {
        "CO2": {
            "type": "co2",
            "input_mode": "emissions",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "CH4": {
            "type": "ch4",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "N2O": {
            "type": "n2o",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
    }

    f = FAIR()
    f.define_time(1850, 1950, 1)
    f.define_scenarios(scenarios)
    f.define_configs(list(range(batch_start, batch_end)))
    f.define_species(species, properties)
    f.allocate()

    # esm-flat10: constant CO2 emissions of 10 GtC/yr for 100 years, no other forcing
    fill(f.emissions, 10 * 44.009 / 12.011, specie="CO2")

    # CH4 and N2O held constant at pre-industrial concentration (see note above)
    fill(f.concentration, 808.2490285, specie="CH4")
    fill(f.concentration, 273.021047, specie="N2O")

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

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], cfg["iirf_0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], cfg["iirf_airborne"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], cfg["iirf_uptake"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], cfg["iirf_temperature"], specie="CO2")

    # forcing scaling
    fill(f.species_configs["forcing_scale"], cfg["scaling_CO2"], specie="CO2")

    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f.species_configs["baseline_concentration"], 284.3169988, specie="CO2")
    fill(
        f.species_configs["forcing_reference_concentration"], 284.3169988, specie="CO2"
    )

    # CH4/N2O baseline = the held-constant concentration above, so their own
    # forcing contribution stays at zero throughout the run
    fill(f.species_configs["baseline_concentration"], 808.2490285, specie="CH4")
    fill(f.species_configs["baseline_concentration"], 273.021047, specie="N2O")
    fill(
        f.species_configs["forcing_reference_concentration"], 808.2490285, specie="CH4"
    )
    fill(
        f.species_configs["forcing_reference_concentration"], 273.021047, specie="N2O"
    )

    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    # temperature at year 100 (cumulative emissions = 1000 GtC by construction) = TCRE
    return f.temperature[100, 0, :, 0]
