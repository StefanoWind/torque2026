# -*- coding: utf-8 -*-
"""
Plot selected SCADA channels
"""
import os
import glob
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.b0')
t_start='2023-08-05T10:00:00'
t_end='2023-08-05T11:59:59'
var_names={'WMET.HorWdSpd':'WindSpeed','WTUR.W':'ActivePower','WNAC.Dir':'Nacelle_Position'}
limit_time=10#[s] max gap filled by interpolation
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbines_sel=['A09','I02','G09']
U_cutin=3#[m/s] cutin wind speed
U_rated=10#[m/s] rated wind speed
U_cutoff=25#[m/s] #cut-off wind speed
wf='King Plains'
vars_sel=['WindSpeed','ActivePower']
vars_sel2=['WindSpeed','ActivePower','Nacelle_Position']
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
dt_ma=60#[s] moving average window

#graphics
colors={'A09':'g','I02':'b','G09':'r'}
labels={'WindSpeed':r'$U_h$ [m s$^{-1}$]','ActivePower':r'$P_{norm}$','Nacelle_Position':'Yaw [$^\circ$]'}
norm={'WindSpeed':1,'ActivePower':2800,'Nacelle_Position':1}

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
scada_all=[]
for file in sorted(glob.glob(os.path.join(source,'*.nc'))):
    scada_ds=xr.open_dataset(file).rename({'WTUR.DateTime':'time'})
    turbine=scada_ds.attrs['Alias']
    scada_ds=scada_ds[list(var_names)].rename(var_names).sel(time=slice(t_start,t_end))

    #irregular (1-7 s) logging interval: bin to a common 1-s grid across turbines and fill short gaps
    scada_ds=scada_ds.resample(time='1s').mean().interpolate_na(dim='time',max_gap=np.timedelta64(limit_time,'s'))
    scada_all.append(scada_ds.expand_dims(turbine=[turbine]))

Data=xr.concat(scada_all,dim='turbine',combine_attrs='drop')
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
    plt.xlim([Data.time[0],Data.time[-1]+np.timedelta64(10,'s')])
    plt.xticks(np.arange(Data.time.values[0],Data.time.values[-1]+np.timedelta64(10,'s'),np.timedelta64(900,'s')))
    ctr+=1
plt.xlabel('Time (UTC)')


ctr=1
plt.figure(figsize=(18,7))
for v in vars_sel2:
    ax=plt.subplot(len(vars_sel2),1,ctr)
    if ctr==1:
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutin,U_cutin],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_rated,U_rated],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutoff,U_cutoff],'--k')
    for t in turbines_sel:
        plt.plot(Data.time,Data[v].sel(turbine=t)/norm[v],alpha=0.25,color=colors[t],linewidth=1)
        plt.plot(Data.time,Data[v].rolling(time=int(dt_ma/dt),center=True).mean().sel(turbine=t)/norm[v],color=colors[t])
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # if ctr<len(vars_sel2):
        # ax.set_xticklabels([])
    plt.grid()
    plt.ylabel(labels[v])
    plt.xlim([Data.time[0],Data.time[-1]+np.timedelta64(1,'s')])
    ctr+=1
plt.xlabel('Time (UTC)')



    
    