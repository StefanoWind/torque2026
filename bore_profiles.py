# -*- coding: utf-8 -*-
"""
Plot profiling data during a bore passage
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import glob
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_trp=os.path.join(cd,'data','awaken','{site}.assist.z01.tropoe.c0','*nc')
source_met=os.path.join(cd,'data','awaken','{site}.met.z01.b0','*nc')
source_uvw=os.path.join('data','awaken','sh.lidar.z02.c0','*nc')
source_str=os.path.join('data','awaken','sgpdlfptS6.b2','*nc')
height_assist=1#[m] height of the ASSIST a.g.l.
sites=['sc1','sb','sg']
site_sel='sg'

#time info
avg_time=10#[min] prebore period
dt=15#[s] timestep
sdate='2023-08-05T10:00:00'#start time
edate='2023-08-05T12:00:00'#end time

#QC
max_gamma=3 #maximumm gamma in TROPoe
max_rmsa=5 #max TROPoe error
min_lwp=5#[g/Kg] min LWP for clouds
limit_height=500#[m] max interpolation gap in height
limit_time=120#[s]  max interpolation gap in time
max_time_diff_trp=60#[s] max time interpolation gap for TROPoe data
max_time_diff_uvw=20#[s] max time interpolation gap for lidar data

#%% Functions
def time_interp(x,time,max_time_diff):
    '''
    Interpolation on new time axis with gap limits
    '''
    tnum=    (time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    tnum_x=(x.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    tnum_x=tnum_x.expand_dims({"height":x.height})
    time_diff=tnum_x.interp(time=time,method="nearest")-tnum
    x_int=x.interp(time=time).where(np.abs(time_diff)<max_time_diff)
    
    return x_int

def interp_nan(x,limit_height,limit_time):
    '''
    Serial nan interpolator on height-time axes
    '''
    x_inp = x.interpolate_na(
            dim="height",
            method="linear",
            max_gap=limit_height).interpolate_na(
            dim="time",
            method="linear",
            max_gap=np.timedelta64(limit_time,'s'))
        
    return x_inp

#%% Initialization
time=np.arange(np.datetime64(sdate),np.datetime64(edate)+np.timedelta64(dt,'s')/2,np.timedelta64(dt,'s'))

#zeroing
theta={}
dtheta={}
r={}
cbh={}

#%% Main

#read wind profiling data
files_uvw=glob.glob(source_uvw)
Data_uvw=xr.open_mfdataset(files_uvw).compute()

#interpolate velocity components
u=interp_nan(time_interp(Data_uvw.u, time, max_time_diff_uvw),limit_height,limit_time)
v=interp_nan(time_interp(Data_uvw.v, time, max_time_diff_uvw),limit_height,limit_time)
w=interp_nan(time_interp(Data_uvw.w, time, max_time_diff_uvw),limit_height,limit_time)

#reconstruct ws/wd
ws=(u**2+v**2)**0.5
wd=(270-np.degrees(np.arctan2(v,u)))%360
real=~np.isnan(wd)
wd=wd.where(wd>10,360).where(real)

#read stare data
files_str=glob.glob(source_str)
Data_str=xr.open_mfdataset(files_str,combine="nested",concat_dim="scanID").compute()
w_str=xr.DataArray(Data_str.wind_speed.where(Data_str.qc_wind_speed==0).values.T.squeeze(),
                   coords={'time':Data_str.time.values.squeeze(),'height':Data_str.range.values}) 

#interpolate w
w_str=interp_nan(time_interp(w_str, time, max_time_diff_uvw),limit_height,limit_time)

#read TROPoe
for s in sites:
    file_trp=glob.glob(source_trp.format(site=s))[0]
    Data_trp=xr.open_dataset(file_trp)
        
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<min_lwp).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=max_gamma
    qc_rmsa=Data_trp['rmsa']<=max_rmsa
    qc_cbh=Data_trp['height']<Data_trp['cbh']
    Data_trp=Data_trp.where(qc_gamma*qc_rmsa*qc_cbh)
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
    #fix height
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
    
    #interpolate thermodynamic quantities
    theta[s]=interp_nan(time_interp(Data_trp.theta,      time, max_time_diff_trp),limit_height,limit_time)
    r[s]=    interp_nan(time_interp(Data_trp.waterVapor, time, max_time_diff_trp),limit_height,limit_time)
    cbh[s]=             time_interp(Data_trp.cbh,        time, max_time_diff_trp)

#read met
Data_met={}
for s in sites:
    files_met=glob.glob(source_met.format(site=s))
    Data_met[s]=xr.open_mfdataset(files_met)
    for v in Data_met[s].data_vars:
        if f'qc_{v}' in Data_met[s].data_vars:
            qc_met=Data_met[s][f'qc_{v}']==0
            Data_met[s][v]=Data_met[s][v].where(qc_met)
            print(f'{np.round(np.sum(qc_met).values/qc_met.size*100,1)}% in {v} retained after filter', flush=True)

#calculate theta difference
for s in sites:
    theta_avg=theta[s].where(theta[s].time<=theta[s].time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')
    dtheta[s]=theta[s]-theta_avg

#%% Plots

#wind maps
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1],width_ratios=[1,0.025])
ax=fig.add_subplot(gs[0,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(ws.time,ws.height,ws.T,np.arange(2,22+.1),cmap='coolwarm',extend='both')
plt.contour(ws.time,ws.height,ws.T,np.arange(2,22+.1),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([Data_trp.time[0],Data_trp.time[-1]])
cax=fig.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$U$ [m s$^{-1}$]')

ax=fig.add_subplot(gs[1,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(wd.time,wd.height,wd.T,np.arange(45,336,10),cmap='gist_stern',extend='both')
plt.contour(wd.time,wd.height,wd.T,np.arange(45,336,10),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([Data_trp.time[0],Data_trp.time[-1]])
cax=fig.add_subplot(gs[1,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [$^\circ$]')

ax=fig.add_subplot(gs[2,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(w.time,w.height,w.T,np.arange(-5,5.1,0.5),cmap='seismic',extend='both')
plt.contour(w.time,w.height,w.T,np.arange(-5,5.1,0.5),colors='k',linewidths=1,alpha=0.25,extend='both')
cf=plt.contourf(w_str.time,w_str.height,w_str.T,np.arange(-5,5.1,0.5),cmap='seismic',extend='both')
plt.contour(w_str.time,w_str.height,w_str.T,np.arange(-5,5.1,0.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
plt.xlabel('Time (UTC)')
plt.xlim([Data_trp.time[0],Data_trp.time[-1]])
cax=fig.add_subplot(gs[2,1])
plt.colorbar(cf,cax=cax,label=r'$w$ [m s$^{-1}$]')
plt.tight_layout()

#theta-r-p
s=site_sel
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1],width_ratios=[1,0.025])
ax=fig.add_subplot(gs[0,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(theta[s].time,theta[s].height,theta[s].T,np.arange(300,315+.1,.5),cmap='hot',extend='both')
plt.contour(theta[s].time,theta[s].height,theta[s].T,np.arange(300,315+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.plot(cbh[s].time,cbh[s]*1000,'ow',markeredgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
cax=fig.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [K]')

ax=fig.add_subplot(gs[1,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),cmap='GnBu',extend='both')
plt.contour(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.plot(cbh[s].time,cbh[s]*1000,'ow',markeredgecolor='k')
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
cax=fig.add_subplot(gs[1,1])
plt.colorbar(cf,cax=cax,label=r'$r$ [g kg$^{-1}$]')

ax=fig.add_subplot(gs[2,0])
plt.plot(Data_met[s].time,Data_met[s].pressure,'k')
plt.xlabel('Time (UTC)')
plt.ylabel('$p$ [kPa]')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim([Data_trp.time[0],Data_trp.time[-1]])
plt.grid()
plt.tight_layout()

#dtheta
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(len(sites), 2, height_ratios=len(sites)*[1],width_ratios=[1,0.025])
ctr=0
for s in sites:
    ax=fig.add_subplot(gs[ctr,0])
    ax.set_facecolor((0.9,0.9,0.9))
    cf=plt.contourf(dtheta[s].time,dtheta[s].height,dtheta[s].T,np.arange(-5,5+.1,0.25),cmap='seismic',extend='both')
    plt.contour(dtheta[s].time,dtheta[s].height,dtheta[s].T,np.arange(-5,5+.1,0.25),colors='k',linewidths=1,alpha=0.25,extend='both')
    plt.plot(cbh[s].time,cbh[s]*1000,'ow',markeredgecolor='k')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlim([Data_trp.time[0],Data_trp.time[-1]])
    plt.ylabel(r'$z$ [m]')
    plt.grid()
    plt.ylim([0,300])
    if ctr<len(sites)-1:
        ax.set_xticklabels([])
    ctr+=1
plt.xlabel('Time (UTC)')
cax=fig.add_subplot(gs[:,1])
plt.colorbar(cf,cax=cax,label=r'$\Delta \theta$ [K]')
plt.tight_layout()

