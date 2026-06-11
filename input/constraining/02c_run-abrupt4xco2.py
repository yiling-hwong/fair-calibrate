#!/usr/bin/env python
# coding: utf-8

"""Run abrupt-4xCO2 runs where RMSE passes"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from parallel_abrupt4xco2 import run_fair
from utils import _parallel_process

if __name__ == "__main__":
    print("Running abrupt-4xCO2 scenarios...")
    load_dotenv()

    batch_size = int(os.getenv("BATCH_SIZE"))
    WORKERS = int(os.getenv("WORKERS"))

    # number of processors
    WORKERS = min(multiprocessing.cpu_count(), WORKERS)

    df_cr = pd.read_csv(
        "../../output/priors/"
        "climate_response_ebm3.csv"
    )

    # we also only want to run ensembles that passed RMSE test
    rmse_pass = np.loadtxt(
        "../../output/posteriors/"
        "runids_rmse_pass.csv"
    ).astype(int)

    # raw 150-year temperature and TOA imbalance series for the Gregory
    # regression, computed as a separate (cheap) step
    n_years = 150
    temp_out = np.ones((n_years, len(rmse_pass))) * np.nan
    toa_out = np.ones((n_years, len(rmse_pass))) * np.nan

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
        temp_out[:, batch_start:batch_end] = res[ibatch][0]
        toa_out[:, batch_start:batch_end] = res[ibatch][1]

    os.makedirs(
        "../../output/prior_runs/",
        exist_ok=True,
    )
    np.save(
        "../../output/prior_runs/"
        "temperature_abrupt-4xCO2_y1-150.npy",
        temp_out,
        allow_pickle=True,
    )
    np.save(
        "../../output/prior_runs/"
        "toa_imbalance_abrupt-4xCO2_y1-150.npy",
        toa_out,
        allow_pickle=True,
    )
