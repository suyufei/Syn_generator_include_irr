# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:20:41 2019

@author: YSu
"""

from __future__ import division
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.api import VAR, DynamicVAR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import random
from datetime import datetime
from datetime import timedelta


sim_data=pd.read_csv('synthetic_irradiance_data.csv',header=0, index_col=0)
sim_data=sim_data.as_matrix()

Normal_Starting=datetime(1900,1,1)

datelist=pd.date_range(Normal_Starting,periods=365)
count=0
m=np.zeros(len(sim_data))
for i in range(0,len(sim_data)):
    m[i]=int(datelist[count].month)
    count= count +1
    if count >364:
        count=0
d_sim=np.column_stack((sim_data,m))   

n=56 
sim_data_yearly= np.reshape(d_sim,(n,365,8))

his_data=pd.read_csv('daily_irr.csv',header=0)
his_data=his_data.as_matrix()

m=np.zeros(len(his_data))
for i in range(0,len(his_data)):
    current_time=his_data[i,0]
    current_time_dt=datetime.strptime(current_time,'%m/%d/%Y')
    m[i]= int(current_time_dt.month)
d_his=np.column_stack((his_data,m))
d_his=d_his[:6935,1:36]
his_data_yearly= np.reshape(d_his,(18,365,8))


for i in range(0,8):
    exec("t%s_ave_sim=np.zeros(12)" %(i))
    exec("t%s_std_sim=np.zeros(12)" %(i))
    exec("t%s_min_sim=np.zeros(12)" %(i))
    exec("t%s_max_sim=np.zeros(12)" %(i))

    exec("t%s_ave_his=np.zeros(12)" %(i))
    exec("t%s_std_his=np.zeros(12)" %(i))
    exec("t%s_min_his=np.zeros(12)" %(i))
    exec("t%s_max_his=np.zeros(12)" %(i))

for i in range(1,13):
    d1=sim_data_yearly[sim_data_yearly[:,:,7]==i]
    d2=his_data_yearly[his_data_yearly[:,:,7]==i]
    for j in range(0,7):
        exec("t%s_ave_sim[i-1]=np.nanmean(d1[:,j])" %(j))
        exec("t%s_std_sim[i-1]=np.nanstd(d1[:,j])" %(j))
        exec("t%s_min_sim[i-1]=np.min(d1[:,j])" %(j))
        exec("t%s_max_sim[i-1]=np.max(d1[:,j])" %(j))
        
        exec("t%s_ave_his[i-1]=np.nanmean(list(d2[:,j]))" %(j))
        exec("t%s_std_his[i-1]=np.nanstd(list(d2[:,j]))" %(j))
        exec("t%s_min_his[i-1]=np.min(list(d2[:,j]))" %(j))
        exec("t%s_max_his[i-1]=np.max(list(d2[:,j]))" %(j))
    
fig_1=plt.figure(1)
plt.title('Temperature Validation')

plt.subplot(331)
plt.errorbar(np.arange(12), t0_ave_sim, t0_std_sim, fmt='bv',lw=3)

#plt.errorbar(np.arange(12), t0_ave_sim, [t0_ave_sim - t0_min_sim, t0_max_sim - t0_ave_sim],
#             fmt='.b', ecolor='grey', lw=1)

plt.errorbar(np.arange(12), t0_ave_his, t0_std_his, fmt='ro',lw=3)
plt.title('Site1')

#plt.errorbar(np.arange(12), t0_ave_his, [t0_ave_his - t0_min_his, t0_max_his - t0_ave_his],
#             fmt='.r', ecolor='orange', lw=1)
plt.xlim(-1, 13)

plt.subplot(332)
plt.errorbar(np.arange(12), t1_ave_sim, t1_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t1_ave_his, t1_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site2')

plt.subplot(333)
plt.errorbar(np.arange(12), t2_ave_sim, t2_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t2_ave_his, t2_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site3')

plt.subplot(334)
plt.errorbar(np.arange(12), t3_ave_sim, t3_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t3_ave_his, t3_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site4')

plt.subplot(335)
plt.errorbar(np.arange(12), t4_ave_sim, t4_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t4_ave_his, t4_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site5')

plt.subplot(336)
plt.errorbar(np.arange(12), t5_ave_sim, t5_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t5_ave_his, t5_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site6')


plt.subplot(337)
plt.errorbar(np.arange(12), t6_ave_sim, t6_std_sim, fmt='bv',lw=3)
plt.errorbar(np.arange(12), t6_ave_his, t6_std_his, fmt='ro',lw=3)
plt.xlim(-1, 13)
plt.title('Site7')