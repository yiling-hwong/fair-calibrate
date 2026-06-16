#!/usr/bin/env python
# coding: utf-8

"""Compute Gregory-regression ECS from abrupt-4xCO2 temperature and TOA imbalance"""

import os

import matplotlib.pyplot as pl
import numpy as np
from dotenv import load_dotenv

pl.switch_backend("agg")

load_dotenv()

plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../defaults.mplstyle")

print("Computing Gregory ECS from abrupt-4xCO2 runs...")

# raw years 1-150 series from 02c, shape (150, len(rmse_pass))
temp = np.load(
    "../../output/prior_runs/"
    "temperature_abrupt-4xCO2_y1-150.npy"
)
toa = np.load(
    "../../output/prior_runs/"
    "toa_imbalance_abrupt-4xCO2_y1-150.npy"
)

# Gregory regression of N (TOA imbalance) on dT (temperature), years 1-150,
# per the AR7 "Method of evaluation": "Gregory regression over first 150
# years of an abrupt-4xCO2 run". Vectorised OLS across all ensemble members.
x_mean = temp.mean(axis=0)
y_mean = toa.mean(axis=0)
slope = ((temp * toa).mean(axis=0) - x_mean * y_mean) / (
    (temp * temp).mean(axis=0) - x_mean**2
)
intercept = y_mean - slope * x_mean

# equilibrium warming where N=0, then halve for 4xCO2 -> 2xCO2 (ECS convention)
dT_eq = -intercept / slope
ecs_gregory = dT_eq / 2

n_unstable = np.sum(slope >= 0)
if n_unstable > 0:
    print(
        f"WARNING: {n_unstable} of {len(slope)} configs have non-negative "
        "Gregory slope (unstable feedback); ecs_gregory will be negative or "
        "infinite for these."
    )

print("ECS (Gregory, 5/50/95):", np.percentile(ecs_gregory, (5, 50, 95)))

os.makedirs(
    "../../output/prior_runs/",
    exist_ok=True,
)
np.save(
    "../../output/prior_runs/"
    "ecs_gregory.npy",
    ecs_gregory,
    allow_pickle=True,
)

if plots:
    rmse_pass = np.loadtxt(
        "../../output/posteriors/"
        "runids_rmse_pass.csv"
    ).astype(np.int64)
    ecs_analytical = np.load(
        "../../output/prior_runs/ecs.npy"
    )[rmse_pass]

    os.makedirs("../../plots/", exist_ok=True)
    fig, ax = pl.subplots()
    ax.scatter(ecs_analytical, ecs_gregory, s=3, alpha=0.3, rasterized=True)
    lims = (
        0,
        max(np.nanmax(ecs_analytical), np.nanmax(ecs_gregory)),
    )
    ax.plot(lims, lims, color="k", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("ECS, analytical (K)")
    ax.set_ylabel("ECS, Gregory 150yr (K)")
    fig.tight_layout()
    pl.savefig(
        "../../plots/"
        "post_rmse_ecs_gregory_vs_analytical.png"
    )
    pl.savefig(
        "../../plots/"
        "post_rmse_ecs_gregory_vs_analytical.pdf"
    )
    pl.close()
