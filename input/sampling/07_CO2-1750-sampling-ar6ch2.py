#!/usr/bin/env python
# coding: utf-8

"""Sampling CO2 concentration in 1750"""

# 1750 concentration is given by the CMIP7 concentration data
# we'll use the same uncertainty range.

import os

import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__

from fair_calibrate.parameters import PRIOR_SAMPLES

load_dotenv()

print("Sampling 1750 CO2 concentration...")

# Get environment variables
load_dotenv()

samples = PRIOR_SAMPLES

# YLH: updated to AR7 Ch2 values (5/50/95 = 276.56/277.60/278.64 ppm, TBC)
# YLH: AR6 was loc=278.377857, scale=2.9/NINETY_TO_ONESIGMA (90% range = 5.8 ppm)
# YLH: AR7 loc=277.60, scale=1.04/NINETY_TO_ONESIGMA (90% range = 2.08 ppm, ~2.8x tighter)
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=samples, loc=277.60, scale=1.04 / NINETY_TO_ONESIGMA, random_state=1067061
)

df = pd.DataFrame({"co2_concentration": co2_1750_conc})

df.to_csv(
    "../../output/priors/"
    "co2_concentration_1750.csv",
    index=False,
)
