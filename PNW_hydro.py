# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:17:00 2018

@author: jkern
"""

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

########################################################
# This script calculates daily hydropower production for 
# PNW zone using FCRPS, Willamette, and missing dams. 

# FCRPS
df_FCRPS = pd.read_csv('PNW_hydro/FCRPS/Modeled_BPAT_dams.csv',header=None)
FCRPS = df_FCRPS.values
F = np.sum(FCRPS,axis=1)

# Willamette

# Missing Dams

df_total = pd.DataFrame(F)
df_total.columns = ['PNW']
df_total.to_excel('PNW_hydro/PNW_hydro_daily.xlsx')

 