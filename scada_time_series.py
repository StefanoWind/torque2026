# -*- coding: utf-8 -*-
"""
Plot selected SCADA channels
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.00')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbines_sel=['A09','I02','G09']
U_cutin=3#[m/s] cutin wind speed
U_rated=10#[m/s] rated wind speed
U_cutoff=25#[m/s] #cut-off wind speed
wf='King Plains'
vars_sel=['WindSpeed','ActivePower']
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
dt_ma=60#[s] moving average window

#graphics
colors={'A09':'g','I02':'b','G09':'r'}
labels={'WindSpeed':r'$U_h$ [m s$^{-1}$]','ActivePower':r'$P_{norm}$'}
norm={'WindSpeed':1,'ActivePower':2800}

#%% Functions
def three_point_star():
    # Points of a 3-pointed star (scaled and centered)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)

#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#zeroing
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
                tn=c.split('Turbine')[-1][:2]
                turbines=np.append(turbines,'0'.join([tn[0],tn[1]]))
                _vars=np.append(_vars,c.split('.')[-1])
    turbines=np.unique(turbines)
    _vars=np.unique(_vars)
    
    #build dataset
    scada_ds=xr.Dataset()
    for v in _vars:
        data=np.zeros((len(scada_df.index),len(turbines)))
        i_t=0
        for t in turbines:
            data[:,i_t]=scada_df[f'PKGP1HIST01.OKWF001_KP_Turbine{t[0]}{t[-1]}.{v}'].values
            i_t+=1
        scada_ds[v]=xr.DataArray(data,coords={'time':scada_df.index.values,'turbine':turbines})
    
    #stack
    if 'time' not in Data.coords:
        Data=scada_ds
    else:
        Data=xr.concat([Data,scada_ds],dim='time')
        
    ctr+=1
dt=np.float64(np.mean(np.diff(Data.time.values)))/10**9

#%% Output
t1=str(Data.time[0].values).replace('T','.').replace(':','').replace('-','')[:-10]
t2=str(Data.time[-1].values).replace('T','.').replace(':','').replace('-','')[:-10]
# Data.to_netcdf(os.path.join(cd,'data',t1+'.'+t2+'.scada.nc'))

#%% Plots
plt.close('all')

ctr=1
plt.figure(figsize=(18,5))
for v in vars_sel:
    ax=plt.subplot(len(vars_sel),1,ctr)
    if ctr==1:
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutin,U_cutin],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_rated,U_rated],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutoff,U_cutoff],'--k')
    for t in turbines_sel:
        plt.plot(Data.time,Data[v].sel(turbine=t)/norm[v],alpha=0.25,color=colors[t],linewidth=1)
        plt.plot(Data.time,Data[v].rolling(time=int(dt_ma/dt),center=True).mean().sel(turbine=t)/norm[v],color=colors[t])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if ctr==1:
        ax.set_xticklabels([])
    plt.grid()
    plt.ylabel(labels[v])
    plt.xlim([Data.time[0],Data.time[-1]])
    ctr+=1
plt.xlabel('Time (UTC)')



    
    