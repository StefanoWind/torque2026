# -*- coding: utf-8 -*-
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pyart
import glob
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pandas as pd
import utm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs

# radar
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
z_rad=100#[m a.g.l.] radar plane
time_shift_rad=120#[s] shift radar time
target_time=np.datetime64('2023-08-05T11:00:00')

# profiling
source_trp=os.path.join(cd,'data','awaken','{site}.assist.z01.tropoe.c0','*nc')
source_uvw=os.path.join(cd,'data','awaken','sh.lidar.z02.c0','*nc')
source_str=os.path.join(cd,'data','awaken','sgpdlfptS6.b2','*nc')
source_blh=os.path.join(cd,'data','pblh_siteG_20230805.100009_20230805.115953.csv')
source_out=os.path.join(cd,'data','siteA1_met_outages_15min_v3.nc')
height_assist=1#[m] height of the ASSIST a.g.l.
sites=['sc1','sb','sg']
site_sel='sg'

#time info
avg_time=10#[min] prebore period
dt=15#[s] timestep
sdate='2023-08-05T10:00:00'
edate='2023-08-05T12:00:00'
sdate_out='2023-08-05T10:00:00'
edate_out='2023-08-06T05:00:00'

#QC
max_gamma=3
max_rmsa=5
min_lwp=5#[g/kg] min LWP for clouds
limit_height=500#[m] max interpolation gap in height
limit_time=120#[s] max interpolation gap in time
max_time_diff_trp=60#[s] max time interpolation gap for TROPoe data
max_time_diff_uvw=20#[s] max time interpolation gap for lidar data

#%% Functions
def time_interp(x,time,max_time_diff):
    tnum=(time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    tnum_x=(x.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    tnum_x=tnum_x.expand_dims({"height":x.height})
    time_diff=tnum_x.interp(time=time,method="nearest")-tnum
    x_int=x.interp(time=time).where(np.abs(time_diff)<max_time_diff)
    return x_int

def interp_nan(x,limit_height,limit_time):
    x_inp = x.interpolate_na(
            dim="height",method="linear",max_gap=limit_height).interpolate_na(
            dim="time",method="linear",max_gap=np.timedelta64(limit_time,'s'))
    return x_inp

#%% Initialization
time=np.arange(np.datetime64(sdate),np.datetime64(edate)+np.timedelta64(dt,'s')/2,np.timedelta64(dt,'s'))

theta={}
r={}
cbh={}

os.makedirs(os.path.join(cd,'figures','ber'),exist_ok=True)

#%% Main

# --- Radar ---
files_rad=glob.glob(source_rad)

# parse radar file timestamps and apply time shift to find closest to target
def parse_rad_time(f):
    base=os.path.basename(f)
    parts=base.split('_')
    return np.datetime64(f'{parts[0][4:8]}-{parts[0][8:10]}-{parts[0][10:12]}T{parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:6]}')+np.timedelta64(time_shift_rad,'s')

rad_times=np.array([parse_rad_time(f) for f in files_rad])
idx_rad=np.argmin(np.abs(rad_times-target_time))
rad_file=files_rad[idx_rad]
rad_time=rad_times[idx_rad]

radar=pyart.io.read_nexrad_archive(rad_file)
gatefilter=pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked("reflectivity")
grid=pyart.map.grid_from_radars(
    (radar,),
    gatefilters=(gatefilter,),
    grid_shape=(1,241,241),
    grid_limits=((z_rad,z_rad),(-250000.0,250000.0),(-250000,250000.0)),
    fields=['reflectivity'])

Z=grid.fields['reflectivity']['data'].squeeze()
xy_rad=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
x_rad=(grid.x['data']+xy_rad[0])/1000
y_rad=(grid.y['data']+xy_rad[1])/1000

# --- Wind ---
files_uvw=glob.glob(source_uvw)
Data_uvw=xr.open_mfdataset(files_uvw).compute()

u=interp_nan(time_interp(Data_uvw.u, time, max_time_diff_uvw),limit_height,limit_time)
v=interp_nan(time_interp(Data_uvw.v, time, max_time_diff_uvw),limit_height,limit_time)
w=interp_nan(time_interp(Data_uvw.w, time, max_time_diff_uvw),limit_height,limit_time)

ws=(u**2+v**2)**0.5
wd=(270-np.degrees(np.arctan2(v,u)))%360
real=~np.isnan(wd)
wd=wd.where(wd>10,360).where(real)

files_str=glob.glob(source_str)
Data_str=xr.open_mfdataset(files_str,combine="nested",concat_dim="scanID").compute()
w_str=xr.DataArray(Data_str.wind_speed.where(Data_str.qc_wind_speed==0).values.T.squeeze(),
                   coords={'time':Data_str.time.values.squeeze(),'height':Data_str.range.values})
w_str=interp_nan(time_interp(w_str, time, max_time_diff_uvw),limit_height,limit_time)

# --- Temperature ---
for s in sites:
    file_trp=glob.glob(source_trp.format(site=s))[0]
    Data_trp=xr.open_dataset(file_trp)
    Data_trp['cbh'][(Data_trp['lwp']<min_lwp).compute()]=Data_trp['height'].max()
    qc_gamma=Data_trp['gamma']<=max_gamma
    qc_rmsa=Data_trp['rmsa']<=max_rmsa
    qc_cbh=Data_trp['height']<Data_trp['cbh']
    Data_trp=Data_trp.where(qc_gamma*qc_rmsa*qc_cbh)
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
    theta[s]=interp_nan(time_interp(Data_trp.theta,      time, max_time_diff_trp),limit_height,limit_time)
    r[s]=    interp_nan(time_interp(Data_trp.waterVapor, time, max_time_diff_trp),limit_height,limit_time)
    cbh[s]=             time_interp(Data_trp.cbh,        time, max_time_diff_trp)

Data_blh=pd.read_csv(source_blh).set_index('Time [UTC]')
Data_blh.index=Data_blh.index.astype('datetime64[ns]')

# --- Outages ---
Data_out=xr.open_dataset(source_out)

#%% Plots

# --- shared data prep ---
X_mesh,Y_mesh=np.meshgrid(x_rad,y_rad)

# Figure 1 — radar contourf on horizontal plane
fig1=plt.figure(figsize=(10,8))
ax1=fig1.add_subplot(111,projection='3d')
cf1=ax1.contourf(X_mesh,Y_mesh,Z,np.arange(-10,51),cmap='Spectral_r',zdir='z',offset=0,extend='both')
ax1.view_init(elev=35,azim=-50)
ax1.set_xlim([x_rad.min(),x_rad.max()])
ax1.set_ylim([y_rad.min(),y_rad.max()])
ax1.set_zlim([0,2000])
ax1.set_axis_off()
ax1.set_title(str(rad_time).replace('T',' ')+' UTC',pad=12)
plt.tight_layout()

# Figure 2 — 5-panel time-height profiles
s=site_sel
fig2=plt.figure(figsize=(18,14))
gs=gridspec.GridSpec(5,2,height_ratios=[1,1,1,1,1],width_ratios=[1,0.025])

ax=fig2.add_subplot(gs[0,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(ws.time,ws.height,ws.T,np.arange(2,22+.1),cmap='coolwarm',extend='both')
plt.contour(ws.time,ws.height,ws.T,np.arange(2,22+.1),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([time[0],time[-1]])
cax=fig2.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$U$ [m s$^{-1}$]')

ax=fig2.add_subplot(gs[1,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(wd.time,wd.height,wd.T,np.arange(45,336,10),cmap='gist_stern',extend='both')
plt.contour(wd.time,wd.height,wd.T,np.arange(45,336,10),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([time[0],time[-1]])
cax=fig2.add_subplot(gs[1,1])
plt.colorbar(cf,cax=cax,label=r'$\gamma$ [$^\circ$]')

ax=fig2.add_subplot(gs[2,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(w.time,w.height,w.T,np.arange(-5,5.1,0.5),cmap='seismic',extend='both')
plt.contour(w.time,w.height,w.T,np.arange(-5,5.1,0.5),colors='k',linewidths=1,alpha=0.25,extend='both')
cf=plt.contourf(w_str.time,w_str.height,w_str.T,np.arange(-5,5.1,0.5),cmap='seismic',extend='both')
plt.contour(w_str.time,w_str.height,w_str.T,np.arange(-5,5.1,0.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([time[0],time[-1]])
cax=fig2.add_subplot(gs[2,1])
plt.colorbar(cf,cax=cax,label=r'$w$ [m s$^{-1}$]')

ax=fig2.add_subplot(gs[3,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(theta[s].time,theta[s].height,theta[s].T,np.arange(300,315+.1,.5),cmap='hot',extend='both')
plt.contour(theta[s].time,theta[s].height,theta[s].T,np.arange(300,315+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.plot(cbh[s].time,cbh[s]*1000,'ob',markeredgecolor='k')
plt.plot(Data_blh.index,Data_blh['BLH (Heffter) [km]']*1000,'ow',markeredgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([time[0],time[-1]])
cax=fig2.add_subplot(gs[3,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [K]')

ax=fig2.add_subplot(gs[4,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),cmap='GnBu',extend='both')
plt.contour(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.plot(cbh[s].time,cbh[s]*1000,'ob',markeredgecolor='k')
plt.plot(Data_blh.index,Data_blh['BLH (Heffter) [km]']*1000,'ow',markeredgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
plt.xlabel('Time (UTC)')
plt.xlim([time[0],time[-1]])
cax=fig2.add_subplot(gs[4,1])
plt.colorbar(cf,cax=cax,label=r'$r$ [g kg$^{-1}$]')

plt.tight_layout()

plt.figure(figsize=(18,5))
plt.plot(Data_out.time,Data_out.outages,color='orange')
plt.fill_between(Data_out.time,Data_out.outages*0,Data_out.outages,color='orange',alpha=0.5)
plt.xlim([np.datetime64(sdate_out),np.datetime64(edate_out)])
plt.ylim([0,2400])
plt.xlabel('Time (UTC)')
plt.ylabel("Customers out")
plt.grid()