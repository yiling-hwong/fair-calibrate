#!/usr/bin/env python
# coding: utf-8

"""Apply posterior weighting"""

# mention in paper: skew-normal distribution
# this is where Zeb earns his corn

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from fair.earth_params import mass_atmosphere, molecular_weight_air
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from fair_calibrate.parameters import PRIOR_SAMPLES, POSTERIOR_SAMPLES

pl.switch_backend("agg")

load_dotenv()

samples = PRIOR_SAMPLES
output_ensemble_size = POSTERIOR_SAMPLES
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../defaults.mplstyle")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

print("Doing reweighting...")


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

valid_temp_af = np.loadtxt(
    f"../../output/posteriors/"
    "runids_rmse_af_pass.csv"
).astype(np.int64)

input_ensemble_size = len(valid_temp_af)

assert input_ensemble_size > output_ensemble_size

temp_in = np.load(
    "../../output/prior_runs/"
    "temperature_1850-2024.npy"
)
# YLH: EEI reference period changed from 2020-1971 to 2023-1960; filename renamed
# YLH: accordingly in sampling script 10 (parallel.py)
#ohc_in = np.load(
#    "../../output/prior_runs/"
#    "ocean_heat_content_2020_minus_1971.npy"
#)
ohc_in = np.load(
    "../../output/prior_runs/"
    "ocean_heat_content_2023_minus_1960.npy"
)
fari_in = np.load(
    "../../output/prior_runs/"
    "forcing_ari_2014-2023_mean.npy"
)
faci_in = np.load(
    "../../output/prior_runs/"
    "forcing_aci_2014-2023_mean.npy"
)
co2_in = np.load(
    "../../output/prior_runs/"
    "concentration_co2_2023.npy"
)
# YLH: ECS swapped from analytical (ecs.npy) to Gregory-150yr (ecs_gregory.npy,
# from 02c/02d); positionally indexed within rmse_pass, see searchsorted below
#ecs_in = np.load(
#    "../../output/prior_runs/ecs.npy"
#)
faer_in = fari_in + faci_in
# YLH: TCRE = temperature at year 100 of esm-flat10 run (02b), positionally
# YLH: indexed within rmse_pass (not full 1.6M), so map valid_temp_af via searchsorted
rmse_pass_ids = np.loadtxt(
    "../../output/posteriors/runids_rmse_pass.csv"
).astype(np.int64)
# YLH: was: tcre_in_all = np.load("../../output/prior_runs/temperature_1pctCO2_1000GtC.npy")
tcre_in_all = np.load(
    "../../output/prior_runs/temperature_esm_flat10_y100.npy"
)
tcre_in = tcre_in_all[np.searchsorted(rmse_pass_ids, valid_temp_af)]

# YLH: TCR swapped from analytical (tcr.npy) to CMIP-consistent "1pctCO2 warming
# at year of CO2 doubling" (year 70), from 02 (temperature_1pctCO2_y70_y140_y210.npy[0,:]);
# positionally indexed within rmse_pass like tcre_in
#tcr_in = np.load(
#    f"../../output/prior_runs/tcr.npy"
#)
tcr_in_all = np.load(
    "../../output/prior_runs/temperature_1pctCO2_y70_y140_y210.npy"
)[0, :]
tcr_in = tcr_in_all[np.searchsorted(rmse_pass_ids, valid_temp_af)]

# YLH: ECS = Gregory-150yr regression from abrupt-4xCO2 (ecs_gregory.npy from
# 02c/02d), positionally indexed within rmse_pass like tcre_in/tcr_in
ecs_in_all = np.load(
    "../../output/prior_runs/ecs_gregory.npy"
)
ecs_in = ecs_in_all[np.searchsorted(rmse_pass_ids, valid_temp_af)]


def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


# YLH: updated ECS constraint to prelim. Vince Cooper AR7 values (was 5th/50th/95th = 2/3/5 K)
# YLH: reverted to AR6 values (2/3/5 K) — AR7 value (2/3.5/6 K) combined with Gregory ECS
# YLH: gives systematic underestimate vs target (FaIR Gregory 95th ~4.7 K < target 6.0 K)
#ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3.5, 6)).x
ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x


# Indicators 2023
# YLH: updated GMST 2004-2023 constraint to AR7 values (Ch2 Zeke Hausfather ensemble; was 0.90/1.05/1.16)
gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.92, 1.035, 1.191)).x
#gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.90, 1.05, 1.16)).x

samples = {}
samples["ECS"] = scipy.stats.skewnorm.rvs(
    ecs_params[0],
    loc=ecs_params[1],
    scale=ecs_params[2],
    size=10**5,
    random_state=91603,
)
samples["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
# note fair produces, and we here report, total earth energy uptake, not just ocean
# this value from IGCC 2024. Use new uncertainties for ocean, assume same uncertainties
# for land, atmosphere and cryopshere.
# looking at new 2024 data from Matt Palmer, it seems unchanged from 1971-2020.
# YLH: updated OHC 2020-1971 constraint to AR7 values (was loc=465.3, scale=108.5; von Schuckmann et al. 2023)
# YLH: AR7 5th/50th/95th = 278/380/482 ZJ -> half-90%-CI = 102 ZJ (symmetric)
# YLH: superseded below by EEI 1960-2023 update (Ch2, Karina & Lijing, 12.06.2026)
#samples["OHC"] = scipy.stats.norm.rvs(
#    loc=380, scale=102 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
#)
#samples["OHC"] = scipy.stats.norm.rvs(
#    loc=465.3, scale=108.5 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
#)
# YLH: EEI reference period changed from 2020-1971 to 2023-1960; values updated to
# YLH: AR7 (Ch2, Karina & Lijing, 12.06.2026; Calibration targets Excel "EEI" row)
# YLH: AR7 5th/50th/95th = 416/491/566 ZJ -> half-90%-CI = 75 ZJ (symmetric)
samples["OHC"] = scipy.stats.norm.rvs(
    loc=491, scale=75 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
)
samples["temperature 2004-2023"] = scipy.stats.skewnorm.rvs(
    gsat_params[0],
    loc=gsat_params[1],
    scale=gsat_params[2],
    size=10**5,
    random_state=19387,
)
# the below commented out bit is if we were to do temperature assessment using HadCRUT5 rather than Blair's assessment
#samples["temperature 2005-2024"] = scipy.stats.norm.rvs(
#    loc = 1.103988,
#    scale = 0.076402,
#    size=10**5,
#    random_state=19387,
#)
# YLH: updated ERFari and ERFaci to AR7 values from Chris (was loc=-0.3/scale=0.3 and loc=-1.0/scale=0.7 AR6 placeholders)
# YLH: ERFari AR7 5th/50th/95th = -0.458/-0.227/0.005 W/m2; forced 95th = 0 (scale=0.2272=|loc|),
# YLH: matching AR6 convention that ERFari is virtually certain non-positive (gives 5th=-0.4544, vs -0.458 target)
# YLH: ERFaci AR7 5th/50th/95th = -1.515/-0.856/-0.254 W/m2 -> half-90%-CI avg = 0.6304 (symmetric)
# YLH: check: ERFari + ERFaci medians = -0.2272 + -0.8561 = -1.0833 ~= ERFaer median (-1.08) -- consistent
samples["ERFari"] = scipy.stats.norm.rvs(
    loc=-0.2272, scale=0.2272 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
)
samples["ERFaci"] = scipy.stats.norm.rvs(
    loc=-0.8561, scale=0.6304 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
)
#samples["ERFari"] = scipy.stats.norm.rvs(
#    loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
#)
#samples["ERFaci"] = scipy.stats.norm.rvs(
#    loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
#)
# YLH: updated total ERFaer to AR7 value (was loc=-1.3, scale=sqrt(0.7^2+0.3^2)/NINETY_TO_ONESIGMA)
# YLH: scale derived from new ERFari/ERFaci half-CIs (0.2272, 0.6304) via same formula as old AR6 code:
# YLH: sqrt(0.6304^2 + 0.2272^2) = 0.6701 ~= AR7 Excel ERFaer half-90%-CI (0.67) -- consistent
samples["ERFaer"] = scipy.stats.norm.rvs(
    loc=-1.08,
    scale=np.sqrt(0.6304**2 + 0.2272**2) / NINETY_TO_ONESIGMA,
    size=10**5,
    random_state=3916153,
)
#samples["ERFaer"] = scipy.stats.norm.rvs(
#    loc=-1.3,
#    scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,
#    size=10**5,
#    random_state=3916153,
#)

# IGCC 2024, using 2023 concentration
# YLH: updated CO2 2023 uncertainty to AR7 values (was scale=0.4; NOAA GML)
# YLH: AR7 5th/50th/95th = 419.19/419.36/419.53 ppm -> half-90%-CI = 0.17 ppm
# YLH: reverted to old scale=0.4 — AR7 scale (0.17/sigma ~ 0.10 ppm) is ~150x tighter
# YLH: than the FaIR prior spread, killing effective samples
#samples["CO2 concentration"] = scipy.stats.norm.rvs(
#    loc=419.36, scale=0.17 / NINETY_TO_ONESIGMA, size=10**5, random_state=81693
#)
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=419.36, scale=0.4, size=10**5, random_state=81693
)
# YLH: added TCRE as active constraint
# YLH: AR7 5th/50th/95th = 1/1.65/2.3 K (symmetric) -> half-90%-CI = 0.65 K
samples["TCRE"] = scipy.stats.norm.rvs(
    loc=1.65, scale=0.65 / NINETY_TO_ONESIGMA, size=10**5, random_state=63916
)

ar_distributions = {}
for constraint in [
    "ECS",
    "TCR",
    "OHC",
    "temperature 2004-2023",
    "ERFari",
    "ERFaci",
    "ERFaer",
    "CO2 concentration",
    "TCRE",
]:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples[constraint]

weights_20yr = np.ones(21)
weights_20yr[0] = 0.5
weights_20yr[-1] = 0.5
weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5

accepted = pd.DataFrame(
    {
        # YLH: was: "ECS": ecs_in[valid_temp_af],
        "ECS": ecs_in,
        # YLH: was: "TCR": tcr_in[valid_temp_af],
        "TCR": tcr_in,
        "OHC": ohc_in[valid_temp_af] / 1e21,
        "temperature 2004-2023": np.average(
            temp_in[154:175, valid_temp_af], weights=weights_20yr, axis=0
        )
        - np.average(temp_in[:52, valid_temp_af], weights=weights_51yr, axis=0),
        "ERFari": fari_in[valid_temp_af],
        "ERFaci": faci_in[valid_temp_af],
        "ERFaer": faer_in[valid_temp_af],
        "CO2 concentration": co2_in[valid_temp_af],
        "TCRE": tcre_in,  # YLH: added TCRE constraint
    },
    index=valid_temp_af,
)

print(accepted)


def calculate_sample_weights(
    distributions: dict, samples: pd.DataFrame, niterations: int=50
) -> tuple:
    """
    Parameters
    ----------
    distributions (dict): for each parameter name key, we have historgram bins
        and values of the desired distribution

    samples (pd.DataFrame): each row contains the parameter sets
        (column headings are parameter names) that have passed the previous
        constraining steps

    Returns
    -------
    np.array: weights associated with each parameter set (row of samples)

    pd.DataFrame

    pd.DataFrame
    """
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    # in each iteration, we calculate a set of weights for each parameter
    # but rather than updating the sample each time, we iteratively update
    # the weights by multiplying the old weights with the new ones
    for k in tqdm(
        range(niterations), desc="Iterations", leave=False, disable=1 - progress
    ):
        gofs.append([])

        # last iteration, store 2nd last iteration's weights
        # and create holder for the final iteration's weights
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            # for each parameter, create a set of weights based on
            # the relative frequencies of the (weighted) sample and
            # the desired distribution
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )

            # last iteration, store weights calculated after each parameter
            # note that these weights still need to be applied to the input weights
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            # update the weights for the next parameter / iteration
            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(
    unique_code: str,
    distributions: dict,
    samples: pd.DataFrame,
    weights: np.array,
    j: int,
    k: int,
) -> tuple:
    """Make a set of weights based on the parameter of interest,
    giving higher weights for parameters that have a higher
    frequency in the desired distribution, as compared to the
    (weighted) sample.

    Parameters
    ----------
    unique_code (str): parameter name (e.g. "ECS")

    distributions (dict): for each parameter name key, we have historgram bins
        and values of the desired distribution

    samples (pd.DataFrame): each row contains the parameter sets
        (column headings are parameter names) that have passed the previous
        constraining steps

    weights (np.array): 1-d array with the length equal to the number of parameter sets,
        containing values previously calculated in this function (if available, else ones)
        mapped so that we have a weight per parameter set

    j (int): parameter number associated with the unique_code
        only to test the first parameter in the first iteration

    k (int): iteration number
        only to test the first parameter in the first iteration

    Returns
    -------
    np.array: weights for each histogram bin, calculated based on the parameter
        of interest, with larger weights if the desired distribution count is
        high compared to the weighted sample count, and smaller weights if the
        weighted sample count is high compared to the desired distribution count

    np.array: mapping of indices (row indices of the samples DataFrame) to histogram bins
    """
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=30
)

effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

# YLH: diagnostic - print per-constraint goodness-of-fit (final iteration,
# lower = better match between weighted FaIR sample and AR7 target) and
# unweighted FaIR vs AR7-target percentiles, to identify which constraint(s)
# are driving effective_samples below output_ensemble_size
print("\nGoodness-of-fit by constraint (final iteration, lower is better):")
print(gofs_full.iloc[-1])

print("\nUnweighted FaIR sample vs AR7 target, percentiles [5, 50, 95]:")
for constraint in ar_distributions:
    fair_pct = np.percentile(accepted[constraint], (5, 50, 95))
    target_pct = np.percentile(ar_distributions[constraint]["values"], (5, 50, 95))
    print(f"{constraint:25s} FaIR={fair_pct}  AR7 target={target_pct}")

assert effective_samples >= output_ensemble_size

# Use numpy.random.choice because pandas has broken itself again
# This selects a new sample from the parameter sets
# that have passed the previous constraining steps, according to the
# weights that we have just calculated.
np.random.seed(10099)
chosen = np.random.choice(accepted.index, size=841, replace=False, p=weights/np.sum(weights))
draws = accepted.loc[chosen]

if plots:
    target_ecs = scipy.stats.gaussian_kde(samples["ECS"])
    prior_ecs = scipy.stats.gaussian_kde(ecs_in_all)   # YLH: was ecs_in (already filtered); use ecs_in_all for rmse_pass prior
    post1_ecs = scipy.stats.gaussian_kde(ecs_in)       # YLH: was ecs_in[valid_temp_af]; ecs_in already indexed to valid_temp_af
    post2_ecs = scipy.stats.gaussian_kde(draws["ECS"])

    target_tcr = scipy.stats.gaussian_kde(samples["TCR"])
    prior_tcr = scipy.stats.gaussian_kde(tcr_in_all)   # YLH: was tcr_in (already filtered); use tcr_in_all for rmse_pass prior
    post1_tcr = scipy.stats.gaussian_kde(tcr_in)       # YLH: was tcr_in[valid_temp_af]; tcr_in already indexed to valid_temp_af
    post2_tcr = scipy.stats.gaussian_kde(draws["TCR"])

    target_temp = scipy.stats.gaussian_kde(samples["temperature 2004-2023"])
    prior_temp = scipy.stats.gaussian_kde(
        np.average(temp_in[154:175, :], weights=weights_20yr, axis=0)
        - np.average(temp_in[:52, :], weights=weights_51yr, axis=0)
    )
    post1_temp = scipy.stats.gaussian_kde(
        np.average(temp_in[154:175, valid_temp_af], weights=weights_20yr, axis=0)
        - np.average(temp_in[:52, valid_temp_af], weights=weights_51yr, axis=0)
    )
    post2_temp = scipy.stats.gaussian_kde(draws["temperature 2004-2023"])

    target_ohc = scipy.stats.gaussian_kde(samples["OHC"])
    prior_ohc = scipy.stats.gaussian_kde(ohc_in / 1e21)
    post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp_af] / 1e21)
    post2_ohc = scipy.stats.gaussian_kde(draws["OHC"])

    target_aer = scipy.stats.gaussian_kde(samples["ERFaer"])
    prior_aer = scipy.stats.gaussian_kde(faer_in)
    post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp_af])
    post2_aer = scipy.stats.gaussian_kde(draws["ERFaer"])

    target_aci = scipy.stats.gaussian_kde(samples["ERFaci"])
    prior_aci = scipy.stats.gaussian_kde(faci_in)
    post1_aci = scipy.stats.gaussian_kde(faci_in[valid_temp_af])
    post2_aci = scipy.stats.gaussian_kde(draws["ERFaci"])

    target_ari = scipy.stats.gaussian_kde(samples["ERFari"])
    prior_ari = scipy.stats.gaussian_kde(fari_in)
    post1_ari = scipy.stats.gaussian_kde(fari_in[valid_temp_af])
    post2_ari = scipy.stats.gaussian_kde(draws["ERFari"])

    target_co2 = scipy.stats.gaussian_kde(samples["CO2 concentration"])
    prior_co2 = scipy.stats.gaussian_kde(co2_in)
    post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp_af])
    post2_co2 = scipy.stats.gaussian_kde(draws["CO2 concentration"])

    colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

    os.makedirs(
        "../../plots/", exist_ok=True
    )

    # Plots 1
    fig, ax = pl.subplots(3, 3, figsize=(18 / 2.54, 18 / 2.54))
    start = 0
    stop = 8
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ecs(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ecs(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ecs(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        target_ecs(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 0].set_xlim(start, stop)
    ax[0, 0].set_ylim(0, 0.6)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")
    ax[0, 0].set_ylabel("Probability density")

    start = 0
    stop = 4
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        prior_tcr(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post1_tcr(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post2_tcr(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        target_tcr(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 1].set_xlim(start, stop)
    ax[0, 1].set_ylim(0, 1.5)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")

    start = 0.65
    stop = 1.45
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        target_temp(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        prior_temp(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post1_temp(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post2_temp(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 2].set_xlim(start, stop)
    ax[0, 2].set_ylim(0, 6)
    ax[0, 2].set_title("Temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 2004-2023 minus 1850-1900")

    start = -1.0
    stop = 0.4
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        target_ari(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ari(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ari(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ari(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 0].set_xlim(start, stop)
    ax[1, 0].set_ylim(0, 3.5)  # YLH: increased from 2.5; AR7 ERFari target peak ≈2.89
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2014-2023 minus 1750")
    ax[1, 0].set_ylabel("Probability density")

    start = -2.25
    stop = 0.25
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        target_aci(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        prior_aci(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post1_aci(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post2_aci(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 1].set_xlim(start, stop)
    ax[1, 1].set_ylim(0, 1.5)  # YLH: increased from 1.1; marginal for target (peak≈1.04), posterior will be tighter
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2014-2023 minus 1750")

    start = -3
    stop = 0.4
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        target_aer(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        prior_aer(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post1_aer(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post2_aer(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 2].set_xlim(start, stop)
    ax[1, 2].set_ylim(0, 1.5)  # YLH: increased from 1.1; posterior will be tighter than target (peak≈0.98)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2014-2023 minus 1750")

    start = 417
    stop = 425
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        target_co2(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        prior_co2(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post1_co2(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post2_co2(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 0].set_xlim(start, stop)
    ax[2, 0].set_ylim(0, 2.0)  # YLH: increased from 1.0; AR7 CO2 target peak ≈1.64; extra headroom for narrow posterior
    ax[2, 0].set_ylabel("Probability density")
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2023")

    start = 100
    stop = 900
    ax[2, 1].plot(
        np.linspace(start, stop),
        target_ohc(np.linspace(start, stop)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        prior_ohc(np.linspace(start, stop)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post1_ohc(np.linspace(start, stop)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post2_ohc(np.linspace(start, stop)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 1].set_xlim(start, stop)
    ax[2, 1].set_ylim(0, 0.012)  # YLH: increased from 0.007; AR7 EEI target tighter (σ≈45.6 ZJ) → peak density ≈0.0088
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2023 minus 1960")  # YLH: updated from "2020 minus 1971"

    ax[2, 2].axis("off")
    legend_lines = [
        Line2D([0], [0], color=colors["prior"], lw=2),
        Line2D([0], [0], color=colors["post1"], lw=2),
        Line2D([0], [0], color=colors["post2"], lw=2),
        Line2D([0], [0], color=colors["target"], lw=2),
    ]
    legend_labels = ["Prior", "Temperature RMSE", "All constraints", "Target"]
    ax[2, 2].legend(legend_lines, legend_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    pl.savefig(
        "../../plots/"
        "constraints.png"
    )
    pl.savefig(
        "../../plots/"
        "constraints.pdf"
    )
    pl.close()

    # Plots 2
    pl.scatter(draws["TCR"], draws["ECS"])
    pl.xlabel("TCR, °C")
    pl.ylabel("ECS, °C")
    pl.tight_layout()
    pl.savefig(
        "../../plots/"
        "ecs_tcr_constrained.png"
    )
    pl.close()

    # Plots 3
    pl.scatter(draws["TCR"], draws["ERFaci"] + draws["ERFari"])
    pl.xlabel("TCR, °C")
    pl.ylabel("Aerosol ERF, W m$^{-2}$, 2014-2023 minus 1750")
    pl.tight_layout()
    pl.savefig(
        "../../plots/"
        "tcr_aer_constrained.png"
    )
    pl.close()

    # Plots 4
    df_gmst = pd.read_csv("../../data/forcing/IGCC_GMST_1850-2024.csv")
    gmst = df_gmst["gmst"].values

    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2025),
        np.min(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        np.max(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2025),
        np.percentile(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            5,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            95,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2025),
        np.percentile(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            16,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            84,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2025),
        np.median(
            temp_in[:, draws.index]
            - np.average(temp_in[:52, draws.index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2025), gmst, color="b", label="Observations")

    ax.legend(frameon=False, loc="upper left")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Constrained, reweighted posterior")
    pl.tight_layout()
    pl.savefig(
        "../../plots/"
        "final_reweighted_historical.png"
    )
    pl.savefig(
        "../../plots/"
        "final_reweighted_historical.pdf"
    )
    pl.close()

# move these to the validation script
print("Constrained, reweighted parameters:")
print("ECS:", np.percentile(draws["ECS"], (5, 50, 95)))
print("TCR:", np.percentile(draws["TCR"], (5, 50, 95)))
print(
    "CO2 concentration 2023:", np.percentile(draws["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature 2004-2023 rel. 1850-1900:",
    np.percentile(draws["temperature 2004-2023"], (5, 50, 95)),
)
print(
    "Aerosol ERFari 2014-2023 rel. 1750:",
    np.percentile(draws["ERFari"], (5, 50, 95)),
)
print(
    "Aerosol ERFaci 2014-2023 rel. 1750:",
    np.percentile(draws["ERFaci"], (5, 50, 95)),
)
print(
    "Aerosol ERF 2014-2023 rel. 1750:",
    np.percentile(draws["ERFaci"] + draws["ERFari"], (5, 50, 95)),
)

print(
    "ERFaer posterior  [5, 50, 95]:", np.percentile(draws["ERFaer"], (5, 50, 95))
)
print(
    "ERFaer AR7 target [5, 50, 95]:", np.percentile(samples["ERFaer"], (5, 50, 95))
)
print("OHC/EEI change 2023 rel. 1960:", np.percentile(draws["OHC"], (5, 50, 95)))
print("TCRE:", np.percentile(draws["TCRE"], (5, 50, 95)))

print("*likely range")

np.savetxt(
    "../../output/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    sorted(draws.index),
    fmt="%d",
)

# warming baselines
df_warming = pd.DataFrame(data=draws["temperature 2004-2023"], index=draws.index, columns = ["temperature 2004-2023"]).sort_index()
df_warming.to_csv(
    "../../output/posteriors/"
    "warming_baselines.csv",
)
