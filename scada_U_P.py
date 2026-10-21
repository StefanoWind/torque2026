# -*- coding: utf-8 -*-
"""
Plot power curve hysteresis
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import glob
from matplotlib.patches import FancyArrowPatch
from scipy import stats
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.b0')
source_all=os.path.join(cd,'data','awaken','kp.turbine.z02.c0')
source_met=os.path.join(cd,'data','awaken','sa1.met.z01.b0','*nc')
turbine_id={'A09':'wt009','I02':'wt073','G09':'wt062'}
turbines_sel=['A09','I02','G09']#selected turbines
t_start='2023-08-05T10:00:00'
t_end='2023-08-05T11:59:59'
limit_time=10#[s] max gap filled by interpolation in high-frequency data
min_flag=0.9#minimum fraction of valid samples in 10-min period

U_rated=11#[m/s] rated wind speed
P_rated=2800#[kW] rated power

dt_ma=60#[s] moving average window
R=287.05#[J/kgK] air gas constant
rho_ref=1.225#[kg/m^3] reference density
bins_U=np.arange(0,26.5,0.5)/U_rated#bins in normalized wind speed
min_grad=10**-3#minimum gradient in U,P
min_N=3#minimum number of 10-min period in bin of power curve [IEC 61400-12-1]

#graphics
skip=20

#%% Functions
def preprocess(ds):
    return ds[["pressure","qc_pressure"]]   # keep only one variable

#%% Load data

#read met data
files_met=glob.glob(source_met)
Data_met = xr.open_mfdataset(
    files_met,
    preprocess=preprocess,
    combine="by_coords",
    parallel=True
)
dt_met=np.float64(np.mean(np.diff(Data_met.time.values)))/10**9
p_met=Data_met.pressure.where(Data_met.qc_pressure==0).rolling(time=int(600/dt_met),center=True).mean()*1000
print('Met data loaded')

#read high-frequency SCADA
scada_all=[]
for tid in turbines_sel:
    files=glob.glob(os.path.join(source,f'*{t_start[:10].replace("-","")}*'+turbine_id[tid]+'*'))
    scada_ds=xr.open_mfdataset(files).rename({'WTUR.DateTime':'time'})
    scada_ds=scada_ds[['WTUR.W','WMET.HorWdSpd','WMET.EnvTmp']].sel(time=slice(t_start,t_end)).compute()

    #irregular (1-7 s) logging interval: bin to a common 1-s grid across turbines and fill short gaps
    scada_ds=scada_ds.resample(time='1s').mean().interpolate_na(dim='time',max_gap=np.timedelta64(limit_time,'s'))
    scada_all.append(scada_ds.expand_dims(turbine=[tid]))
Data=xr.concat(scada_all,dim='turbine',combine_attrs='drop')
dt=np.float64(np.mean(np.diff(Data.time.values)))/10**9
p=p_met.interp(time=Data.time)
T=Data['WMET.EnvTmp']+273.15
rho=p/(T*R)
P=Data['WTUR.W'].rolling(time=int(dt_ma/dt),center=True).mean()/P_rated
U=(Data['WMET.HorWdSpd']*(rho/rho_ref)**(1/3)/U_rated).rolling(time=int(dt_ma/dt),center=True).mean()
print('High-frequency SCADA loaded')

#read all SCADA
P_all={}
U_all={}
for tid in turbines_sel:
    files=glob.glob(os.path.join(source_all,'*'+turbine_id[tid]+'*'))
    print(f'{len(files)} files available for {tid}')
    Data_all=xr.open_mfdataset(files).rename({'WTUR.DateTime':'time'})
    Data_all=Data_all.where(Data_all['Total_Flag_10m_Percent']>=min_flag)
    p_all=p_met.interp(time=Data_all.time)
    T_all=Data_all['WMET.EnvTmp_10m_Avg']+273.15
    rho_all=p_all/(T_all*R)
    P_all[tid]=Data_all['WTUR.W_10m_Avg'].compute()/P_rated
    U_all[tid]=(Data_all['WMET.HorWdSpd_10m_Avg']*(rho_all/rho_ref)**(1/3)).compute()/U_rated
    print(f'Data of {tid} loaded')

#%% Power curves
U_avg={}
P_avg={}
P_std={}
N={}
for tid in turbines_sel:
    nan=np.isnan(U_all[tid].values+P_all[tid].values)
    flat=(np.abs(np.gradient(U_all[tid].values))<min_grad)*(np.abs(np.gradient(P_all[tid].values))<min_grad)
    sel=nan+flat==0
    U_avg[tid]=stats.binned_statistic(U_all[tid].values[sel], U_all[tid].values[sel],bins=bins_U,statistic='mean')[0]
    P_avg[tid]=stats.binned_statistic(U_all[tid].values[sel], P_all[tid].values[sel],bins=bins_U,statistic='mean')[0]
    N[tid]=    stats.binned_statistic(U_all[tid].values[sel], P_all[tid].values[sel],bins=bins_U,statistic='count')[0]
    
    U_avg[tid][N[tid]<min_N]=np.nan
    P_avg[tid][N[tid]<min_N]=np.nan
    U_avg[tid][np.isnan(U_avg[tid])]=(bins_U[:-1]+bins_U[1:])[np.isnan(U_avg[tid])]/2
    P_avg[tid][U_avg[tid]>1.1]=1
    
    print(f'Power curve of {tid} calculated excluding {np.sum(flat)} flat points')


#%% Plots
plt.close('all')

ctr=0
dtime=(Data.time.values-Data.time.values[0])/np.timedelta64(60,'s')
fig=plt.figure(figsize=(18,5))
gs = gridspec.GridSpec(1, len(turbines_sel)+1,width_ratios=[1]*len(turbines_sel)+[0.05])
for tid in turbines_sel:
    ax=fig.add_subplot(gs[ctr])
    plt.fill_between(U_avg[tid],np.zeros(len(P_avg[tid]))-0.1, P_avg[tid], color=(0,0,0,0.5))
    x=U.sel(turbine=tid).values
    y=P.sel(turbine=tid).values
    
    plt.plot(U.sel(turbine=tid),P.sel(turbine=tid),'k',linewidth=1)
    for i in range(0, len(x)-1, skip):
        if ~np.isnan(x[i]+y[i]):
            arrow = FancyArrowPatch((x[i], y[i]), (x[i+1], y[i+1]),
                                    arrowstyle='->', color='k', mutation_scale=15)
            ax.add_patch(arrow)
    
    sc=plt.scatter(x[::skip],y[::skip],s=5,c=dtime[::skip],cmap='jet',zorder=10,vmin=0,vmax=120)
    plt.xlabel(r'$U_{norm}$')
    if ctr==0:
        plt.ylabel(r'$P_{norm}$')
    else:
        ax.set_yticklabels([])
    plt.grid()
    plt.xlim([0,2.25])
    plt.ylim([-0.01,1.05])
    ctr+=1
cax=fig.add_subplot(gs[-1])
plt.colorbar(sc,cax,label=f'Time since {str(Data.time.values[0])[11:16]} UTC [min]',ticks=np.arange(0,121,30))
    
    