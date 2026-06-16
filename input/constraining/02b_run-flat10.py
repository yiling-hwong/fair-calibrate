#!/usr/bin/env python
# coding: utf-8

"""Run esm-flat10 runs where RMSE passes"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from parallel_flat10 import run_fair
from utils import _parallel_process

if __name__ == "__main__":
    print("Running esm-flat10 scenarios...")
    load_dotenv()

    batch_size = int(os.getenv("BATCH_SIZE"))
    WORKERS = int(os.getenv("WORKERS"))

    # number of processors
    WORKERS = min(multiprocessing.cpu_count(), WORKERS)

    df_cc = pd.read_csv(
        "../../output/priors/"
        "carbon_cycle.csv"
    )
    df_cr = pd.read_csv(
        "../../output/priors/"
        "climate_response_ebm3.csv"
    )
    df_scaling = pd.read_csv(
        "../../output/priors/"
        "forcing_scaling.csv"
    )

    # we also only want to run ensembles that passed RMSE test
    rmse_pass = np.loadtxt(
        "../../output/posteriors/"
        "runids_rmse_pass.csv"
    ).astype(int)

    # we only care about temperature at year 100 (= TCRE, since cumulative
    # emissions are 1000 GtC by construction)
    temp_y100_out = np.ones(len(rmse_pass)) * np.nan

    config = []
    for ibatch, batch_start in enumerate(range(0, len(rmse_pass), batch_size)):
        config.append({})
        batch_end = min(batch_start + batch_size, len(rmse_pass))
        config[ibatch]["batch_start"] = batch_start
        config[ibatch]["batch_end"] = batch_end
        config[ibatch]["c1"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c1"].values
        config[ibatch]["c2"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c2"].values
        config[ibatch]["c3"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c3"].values
        config[ibatch]["kappa1"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa1"
        ].values
        config[ibatch]["kappa2"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa2"
        ].values
        config[ibatch]["kappa3"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa3"
        ].values
        config[ibatch]["epsilon"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "epsilon"
        ].values
        config[ibatch]["gamma"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "gamma"
        ].values
        config[ibatch]["forcing_4co2"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "F_4xCO2"
        ].values
        config[ibatch]["iirf_0"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "r0"
        ].values.squeeze()
        config[ibatch]["iirf_airborne"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rA"
        ].values.squeeze()
        config[ibatch]["iirf_uptake"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rU"
        ].values.squeeze()
        config[ibatch]["iirf_temperature"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rT"
        ].values.squeeze()
        config[ibatch]["scaling_CO2"] = df_scaling.loc[
            rmse_pass[batch_start:batch_end], "CO2"
        ].values.squeeze()

    parallel_process_kwargs = dict(
        func=run_fair,
        configuration=config,
        config_are_kwargs=False,
    )

    with ProcessPoolExecutor(WORKERS) as pool:
        res = _parallel_process(
            **parallel_process_kwargs,
            pool=pool,
        )

    for ibatch, batch_start in enumerate(range(0, len(rmse_pass), batch_size)):
        batch_end = min(batch_start + batch_size, len(rmse_pass))
        temp_y100_out[batch_start:batch_end] = res[ibatch]

    os.makedirs(
        "../../output/prior_runs/",
        exist_ok=True,
    )
    np.save(
        "../../output/prior_runs/"
        "temperature_esm_flat10_y100.npy",
        temp_y100_out,
        allow_pickle=True,
    )
