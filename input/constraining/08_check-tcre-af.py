#!/usr/bin/env python
# coding: utf-8

"""Spit out the TCRE and airborne fraction."""

# we don't constrain on these as they are model-based assessments, but we want to add
# to the table.

import os

import numpy as np
from dotenv import load_dotenv
from fair.earth_params import mass_atmosphere, molecular_weight_air

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

af = np.load(
    f"../../output/"
    "prior_runs/airborne_fraction_1pctCO2_y70_y140_y210.npy"
)
temp = np.load(
    f"../../output/"
    "prior_runs/temperature_1pctCO2_y70_y140_y210.npy"
)
# YLH: old TCRE proxy from 1pctCO2 run (kept for comparison); AR7 uses esm-flat10 below
temp1000 = np.load(
    f"../../output/"
    "prior_runs/temperature_1pctCO2_1000GtC.npy"
)
# YLH: AR7 TCRE source: temperature at year 100 of esm-flat10 run (02b), cumulative
# YLH: emissions = 1000 GtC at year 100; indexed within rmse_pass (same as pass1 below)
temp_flat10 = np.load(
    "../../output/"
    "prior_runs/temperature_esm_flat10_y100.npy"
)
pass1 = np.loadtxt(
    f"../../output/"
    "posteriors/runids_rmse_pass.csv",
    dtype=int,
)
pass2 = np.loadtxt(
    f"../../output/"
    "posteriors/runids_rmse_reweighted_pass.csv",
    dtype=int,
)

co2_1850 = 284.3169988
co2_1920 = co2_1850 * 1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
mass_factor = 12.011 / molecular_weight_air * mass_atmosphere / 1e21
# mass_factor converts ppm CO2 to (1000 Gt C)

idx = np.isin(pass1, pass2).nonzero()[0]
print("temperature 2xCO2:", np.percentile(temp[0, idx], (5, 50, 95)))
print("temperature 4xCO2:", np.percentile(temp[1, idx], (5, 50, 95)))
print("temperature 8xCO2:", np.percentile(temp[2, idx], (5, 50, 95)))
# YLH: old 1pctCO2 proxy kept for reference
print("TCRE @1000GtC (1pctCO2 proxy):", np.percentile(temp1000[idx], (5, 50, 95)))
# YLH: AR7 constraint source — esm-flat10 y100; idx is positional within rmse_pass = pass1
print("TCRE @1000GtC (esm-flat10, AR7):", np.percentile(temp_flat10[idx], (5, 50, 95)))
print("AF 2xCO2*:", np.percentile(af[0, idx], (16, 50, 84)))
print("AF 4xCO2*:", np.percentile(af[1, idx], (16, 50, 84)))
print("AF 8xCO2*:", np.percentile(af[2, idx], (16, 50, 84)))
print(
    "TCRE (IPCC method)*:",
    np.percentile(
        af[0, idx] * temp[0, idx] / ((co2_1920 - co2_1850) * mass_factor), (16, 50, 84)
    ),
)
print("*likely range")
