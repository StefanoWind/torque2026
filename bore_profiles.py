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
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import glob
from scipy.interpolate import make_smoothing_spline
from scipy.signal import butter, sosfiltfilt
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
source_uvw=os.path.join(cd,'data','awaken','sh.lidar.z02.c0','*nc')
source_str=os.path.join(cd,'data','awaken','sgpdlfptS6.b2','*nc')
source_blh=os.path.join(cd,'data','pblh_siteG_20230805.100009_20230805.115953.csv')
height_assist=1#[m] height of the ASSIST a.g.l.
sites=['sc1','sb','sg']
site_sel='sg'

#time info
avg_time=10#[min] prebore period
dt=15#[s] timestep
sdate='2023-08-05T10:00:00'#start time
edate='2023-08-05T12:00:00'#end time
prebore=['2023-08-05T10:00:00','2023-08-05T10:20:00']
g=9.81#[m/s^2] gravity acceleration
lam_U=1000#smoothing penalty of spline fit of pre-bore U (None=GCV)
max_height_fit_U=2000#[m] max height of spline fit of pre-bore U
fluc_period=['2023-08-05T11:00:00','2023-08-05T11:20:00']#period for u', w' fluctuations
band_period=[2,12]#[min] min/max wave period of band-pass filter
order_filt=4#order of Butterworth filter
min_valid_filt=0.8#minimum fraction of valid samples in time to band-pass a height

met_height=2#[m] height of temperature measurements at AWAKEN

#QC
max_gamma=3 #maximumm gamma in TROPoe
max_rmsa=5 #max TROPoe error
min_lwp=5#[g/Kg] min LWP for clouds
limit_height=500#[m] max interpolation gap in height
limit_time=120#[s]  max interpolation gap in time
max_time_diff_trp=60#[s] max time interpolation gap for TROPoe data
max_time_diff_uvw=20#[s] max time interpolation gap for lidar data

#graphics
colors={'sc1':'m','sb':'c','sg':'orange'}
labels={'sc1':'Site C1a','sb':'Site B','sg':'Site G'}

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

def bandpass(x,band_period,dt,order,min_valid):
    '''
    Zero-phase Butterworth band-pass in time [Butterworth, 1930; Gustafsson, 1996]
    '''
    sos=butter(order,[1/(band_period[1]*60),1/(band_period[0]*60)],btype='bandpass',fs=1/dt,output='sos')

    #the filter propagates any NaN to the whole series, so gaps are filled before filtering and re-masked after
    valid=~np.isnan(x)
    x_fill=x.interpolate_na(dim='time',method='linear').ffill(dim='time').bfill(dim='time')
    x_filt=xr.apply_ufunc(lambda y: sosfiltfilt(sos,y,axis=-1),x_fill,input_core_dims=[['time']],output_core_dims=[['time']])

    return x_filt.transpose(*x.dims).where(valid).where(valid.mean(dim='time')>=min_valid)

#%% Initialization
time=np.arange(np.datetime64(sdate),np.datetime64(edate)+np.timedelta64(dt,'s')/2,np.timedelta64(dt,'s'))

#zeroing
T={}
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
    T[s]=    interp_nan(time_interp(Data_trp.temperature,time, max_time_diff_trp),limit_height,limit_time)
    theta[s]=interp_nan(time_interp(Data_trp.theta,      time, max_time_diff_trp),limit_height,limit_time)
    r[s]=    interp_nan(time_interp(Data_trp.waterVapor, time, max_time_diff_trp),limit_height,limit_time)
    cbh[s]=             time_interp(Data_trp.cbh,        time, max_time_diff_trp)

#read met
Data_met={}
for s in sites:
    files_met=glob.glob(source_met.format(site=s))
    Data_met[s]=xr.open_mfdataset(files_met)
    for var in Data_met[s].data_vars:
        if f'qc_{var}' in Data_met[s].data_vars:
            qc_met=Data_met[s][f'qc_{var}']==0
            Data_met[s][var]=Data_met[s][var].where(qc_met)
            print(f'{np.round(np.sum(qc_met).values/qc_met.size*100,1)}% in {var} retained after filter', flush=True)

#calculate theta difference
for s in sites:
    theta_avg=theta[s].where(theta[s].time<=theta[s].time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')
    dtheta[s]=theta[s]-theta_avg
    
#read PBLH
Data_blh=pd.read_csv(source_blh).set_index('Time [UTC]')
Data_blh.index = Data_blh.index.astype('datetime64[ns]')


#pre-bore profiles
U_pre=Data_uvw.u.sel(time=slice(prebore[0],prebore[1])).mean(dim='time')
V_pre=Data_uvw.v.sel(time=slice(prebore[0],prebore[1])).mean(dim='time')
theta_pre=Data_trp.theta.sel(time=slice(prebore[0],prebore[1])).mean(dim='time')

bore_direction=310
C_bore=20 
U_scorer=U_pre*np.cos(np.deg2rad(270-bore_direction))+V_pre*np.sin(np.deg2rad(270-bore_direction))
N_scorer=(g/theta_pre*theta_pre.differentiate('height')).interp(height=U_scorer.height)

#curvature from cubic smoothing spline with GCV penalty to avoid noise amplification of finite differences [Reinsch, 1967; Craven and Wahba, 1979]
fit=~np.isnan(U_scorer)&(U_scorer.height<=max_height_fit_U)
z_fit=U_scorer.height[fit]
spl_U=make_smoothing_spline(z_fit.values,U_scorer[fit].values,lam=lam_U)
d2U_dz_scorer=xr.DataArray(spl_U.derivative(2)(U_scorer.height.values),coords={'height':U_scorer.height}).where((U_scorer.height>=z_fit.min())&(U_scorer.height<=z_fit.max()))

l_scorer=N_scorer/(U_scorer-C_bore)**2-d2U_dz_scorer/(U_scorer-C_bore)

#wave fluctuations: along-bore u and w band-passed identically, padded by the longest period to limit filter edge effects
pad=np.timedelta64(band_period[1]*60,'s')
sel_pad=slice(np.datetime64(fluc_period[0])-pad,np.datetime64(fluc_period[1])+pad)
u_bore=u*np.cos(np.deg2rad(270-bore_direction))+v*np.sin(np.deg2rad(270-bore_direction))
u_fluc=u_bore.sel(time=sel_pad)-u_bore.sel(time=sel_pad).mean(dim='time')
w_fluc=w.sel(time=sel_pad)-w.sel(time=sel_pad).mean(dim='time')

#%% Plots
plt.close('all')

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
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
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
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
cax=fig.add_subplot(gs[1,1])
plt.colorbar(cf,cax=cax,label=r'$\gamma$ [$^\circ$]')

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
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
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
plt.plot(cbh[s].time,cbh[s]*1000,'ob',markeredgecolor='k')
plt.plot(Data_blh.index,Data_blh['BLH (Heffter) [km]']*1000,'ow',markeredgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
plt.ylim([0,2000])
ax.set_xticklabels([])
cax=fig.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [K]')

ax=fig.add_subplot(gs[1,0])
ax.set_facecolor((0.9,0.9,0.9))
cf=plt.contourf(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),cmap='GnBu',extend='both')
plt.contour(r[s].time,r[s].height,r[s].T,np.arange(7,17+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.plot(cbh[s].time,cbh[s]*1000,'ob',markeredgecolor='k')
plt.plot(Data_blh.index,Data_blh['BLH (Heffter) [km]']*1000,'ow',markeredgecolor='k')
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
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
plt.grid()
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
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
    plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
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


#time history of T
plt.figure(figsize=(18,6))
for s in sites:
    plt.plot(Data_met[s].time,Data_met[s].temperature,color=colors[s],linewidth=2,label=labels[s])
    plt.plot(T[s].time,T[s].interp(height=met_height),'o',color=colors[s],markersize=5,markeredgecolor='k')

plt.plot(Data_trp.time[0]-np.timedelta64(10,'s'),100,'-k',linewidth=2,label='Met station')
plt.plot(Data_trp.time[0]-np.timedelta64(10,'s'),100,'ok',label='TROPoe')
plt.xlim([Data_trp.time[0]-np.timedelta64(10,'s'),Data_trp.time[-1]+np.timedelta64(10,'s')])
plt.ylim([24,28])
plt.ylabel(r'$T$ [$^\circ$C]')
plt.xlabel('Time (UTC)')
plt.legend()
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

#pre-bore U along bore direction: raw vs spline fit
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(U_scorer,U_scorer.height,'.-k',label='Raw')
plt.plot(spl_U(z_fit.values),z_fit,'-r',linewidth=2,label='Smoothing spline')
plt.axvline(C_bore,color='b',linestyle='--',label=r'$c$')
plt.xlabel(r'$U$ along bore direction [m s$^{-1}$]')
plt.ylabel(r'$z$ [m]')
plt.ylim([0,max_height_fit_U])
plt.legend()
plt.grid()
plt.tight_layout()

#Scorer parameter
plt.subplot(1,2,2)
plt.plot(N_scorer/(U_scorer-C_bore)**2,l_scorer.height,'-b',label=r'$N^2/(U-c)^2$')
plt.plot(-d2U_dz_scorer/(U_scorer-C_bore),l_scorer.height,'-r',label=r'$-U^{\prime\prime}/(U-c)$')
plt.plot(l_scorer,l_scorer.height,'.-k',label=r'$l^2$')
plt.axvline(0,color='gray',linestyle='--')
plt.xlabel(r'$l^2$ [m$^{-2}$]')
plt.ylabel(r'$z$ [m]')
plt.ylim([0,max_height_fit_U])
plt.legend()
plt.grid()
plt.tight_layout()

#band-passed fluctuations
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(2, 2, height_ratios=[1,1],width_ratios=[1,0.025])
for i,(f,label) in enumerate(zip([u_fluc,w_fluc],[r"$u'$ [m s$^{-1}$]",r"$w'$ [m s$^{-1}$]"])):
    ax=fig.add_subplot(gs[i,0])
    ax.set_facecolor((0.9,0.9,0.9))
    cf=plt.contourf(f.time,f.height,f.T,np.arange(-5*1.25,5.1*1.25,0.25),cmap='seismic',extend='both')
    plt.contour(f.time,f.height,f.T,np.arange(-5,5.1*1.25,0.25),colors='k',linewidths=1,alpha=0.25,extend='both')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel(r'$z$ [m]')
    plt.grid()
    plt.ylim([0,2000])
    if i==0:
        ax.set_xticklabels([])
    else:
        plt.xlabel('Time (UTC)')
    cax=fig.add_subplot(gs[i,1])
    plt.colorbar(cf,cax=cax,label=label)
plt.tight_layout()

