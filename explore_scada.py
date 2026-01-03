# -*- coding: utf-8 -*-
"""
Plot all SCADA channels
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import glob
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.00')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbines_sel=['A9','B1','C1','D1','E1','F1','G1','H1','I1','J1']
turbines_sel=['A9','I2','G9','E6']
wf='King Plains'

#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
Data=xr.Dataset()

#%% Main

#read scada
ctr=0
for file in os.listdir(source):
    scada_df=pd.read_csv(os.path.join(source,file))
    scada_df=scada_df.rename(columns={scada_df.columns[0]: "time"}).set_index('time')
    scada_df.index= pd.to_datetime(scada_df.index, utc=True).tz_convert(None)
    
    #extract turbine and variable list
    if ctr==0:
        _vars=[]
        turbines=[]
        for c in scada_df.columns:
            if 'Turbine' in c:
                # scada_df=scada_df.rename(columns={c: c.split('Turbine')[-1]})
                turbines=np.append(turbines,c.split('Turbine')[-1][:2])
                _vars=np.append(_vars,c.split('.')[-1])
    turbines=np.unique(turbines)
    _vars=np.unique(_vars)
    
    #build dataset
    scada_ds=xr.Dataset()
    for v in _vars:
        data=np.zeros((len(scada_df.index),len(turbines)))
        i_t=0
        for t in turbines:
            data[:,i_t]=scada_df[f'PKGP1HIST01.OKWF001_KP_Turbine{t}.{v}'].values
            i_t+=1
        scada_ds[v]=xr.DataArray(data,coords={'time':scada_df.index.values,'turbine':turbines})
    
    #stack
    if 'time' not in Data.coords:
        Data=scada_ds
    else:
        Data=xr.concat([Data,scada_ds],dim='time')
        
    ctr+=1
    
    
#%% Plots
plt.close('all')
# for v in Data.data_vars:
#     plt.figure(figsize=(18,5))
#     for t in turbines_sel:
#         plt.plot(Data.time,Data[v].sel(turbine=t),label=t)
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     plt.grid()
#     plt.xlabel('Time (UTC)')
#     plt.ylabel(v)
# plt.legend()

for v in Data.data_vars:
    plt.figure(figsize=(18,5))
    for t in turbines_sel:
        lp,=plt.plot(Data.time,Data[v].sel(turbine=t),alpha=0.1)
        color = lp.get_color()
        plt.plot(Data.time,Data[v].rolling(time=120,center=True).mean().sel(turbine=t),label=t,color=color)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid()
    plt.xlabel('Time (UTC)')
    plt.ylabel(v)
plt.legend()

plt.figure()
x_tur=Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
y_tur=Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
plt.scatter(x_tur,y_tur,s=10,c='k')
for t in turbines_sel:
    x_tur=Turbines.x_utm.where(Turbines.name==f'{t[0]}0{t[1]}',drop=True).values/1000
    y_tur=Turbines.y_utm.where(Turbines.name==f'{t[0]}0{t[1]}',drop=True).values/1000
    plt.plot(x_tur,y_tur,'.',markersize=10,label=t)
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.xlim([610,650])
plt.ylim([4010,4037.5])
plt.gca().set_aspect('equal')
plt.legend()