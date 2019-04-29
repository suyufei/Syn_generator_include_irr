# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:00:52 2019

@author: Joy Hill
"""

#ICF calculation

import pandas as pd
import numpy as np


d=pd.read_csv('Synthetic_streamflows/synthetic_streamflows_TDA.csv',header=None)
doy = np.arange(1,366)
doy_array = np.tile(doy,int(len(d)/365))
doy_array = pd.DataFrame(doy_array)
d = np.array(pd.concat([doy_array,d],axis=1))
d = d[243:len(d)-122,:]
years = int(len(d)/365)
ICFs = np.zeros((years,1))

for i in range(0,years):
    
    j = d[i*365:i*365+365,0]
    a = d[i*365:i*365+365,1]

    b = np.argwhere(a>450000)
    if len(b) > 0:
        
        ICFs[i] = min(j[b]*(j[b]>80)) 
    
    
    else:
        ICFs[i] = 0
        
    

    

np.savetxt('PNW_hydro/FCRPS/ICFcal.csv',ICFs,delimiter=',')