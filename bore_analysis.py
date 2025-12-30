# -*- coding: utf-8 -*-
"""
Plot data during a bore passage
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import scipy as sp
import glob
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
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

avg_time=10#[min]

#QC
max_gamma=3 #maximumm gamma in TROPoe
max_rmsa=5 #max TROPoe error
min_lwp=5#[g/Kg] min LWP for clouds
window=60#[s]
max_mad_T=1
max_mad_r=1
max_mad_ws=2
max_mad_wd=10
max_mad_w=5
limit_trp=1
limit_uvw=5

#graphics
T_min=300
T_max=315
r_min=7
r_max=17

#%% Functions
def interp_nan(x,limit):
    valid_mask1 = ~np.isnan(x)
    distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
    interp_mask1 = (np.isnan(x)) & (distance1 <= limit)
    yy1, xx1 = np.indices(x.shape)
    points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
    values1 = x.values[valid_mask1]
    interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
    interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
    ws1_inpaint = x.values.copy()
    ws1_inpaint[interp_mask1] = interpolated_values1
    return xr.DataArray(ws1_inpaint,coords=x.coords)

def mad_despike(x,window,max_mad):
    dt=np.nanmedian(np.diff(x.time.values)/np.timedelta64(1,'s'))
    ma_center=x.rolling(time=int(int(window/dt)/2)*2+1, center=True).median()  
    ma_left  = x.rolling(time=int(int(window/dt)/2)*2+1, center=False).median()
    ma_right = x[::-1].rolling(time=int(int(window/dt)/2)*2+1, center=False).median()[::-1] 
    ma=ma_center.fillna(ma_left).fillna(ma_right)
    mad=x-ma
    qc=np.abs(mad)<max_mad
    x_dsp=x.where(qc)
    print(f'{np.round(np.sum(qc).values/qc.size*100,1)}% retained in {x.name} after spike filter', flush=True)

    return x_dsp
   
#%% Initialization
Data_trp={}

#read TROPoe
for s in sites:
    file_trp=glob.glob(source_trp.format(site=s))[0]
    Data_trp[s]=xr.open_dataset(file_trp)
        
    #qc tropoe data
    Data_trp[s]['cbh'][(Data_trp[s]['lwp']<min_lwp).compute()]=Data_trp[s]['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp[s]['gamma']<=max_gamma
    qc_rmsa=Data_trp[s]['rmsa']<=max_rmsa
    qc_cbh=Data_trp[s]['height']<Data_trp[s]['cbh']
    Data_trp[s]=Data_trp[s].where(qc_gamma*qc_rmsa*qc_cbh)
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
    Data_trp[s]['theta']=interp_nan(mad_despike(Data_trp[s]['theta'], window, max_mad_ws),limit_trp)
    Data_trp[s]['waterVapor']=interp_nan(mad_despike(Data_trp[s]['waterVapor'], window, max_mad_ws),limit_trp)
   
    #fix coordinates
    Data_trp[s]=Data_trp[s].assign_coords(height=Data_trp[s].height*1000+height_assist)
    Data_trp[s]=Data_trp[s].resample(time='20s').nearest(tolerance='60s')     

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
      
#read lidar
files_uvw=glob.glob(source_uvw)
Data_uvw=xr.open_mfdataset(files_uvw)
Data_uvw=Data_uvw.resample(time='10s').nearest(tolerance='5s') 

Data_uvw['u']=interp_nan(mad_despike(Data_uvw.u, window, max_mad_ws),limit_uvw)
Data_uvw['v']=interp_nan(mad_despike(Data_uvw.v, window, max_mad_ws),limit_uvw)
Data_uvw['w']=interp_nan(mad_despike(Data_uvw.w, window, max_mad_ws),limit_uvw)

Data_uvw['ws']=(Data_uvw['u']**2+Data_uvw['v']**2)**0.5
Data_uvw['wd']=(270-np.degrees(np.arctan2(Data_uvw['v'],Data_uvw['u'])))%360
real=~np.isnan(Data_uvw['wd'])
Data_uvw['wd']=Data_uvw['wd'].where(Data_uvw['wd']>10,360).where(real)

files_str=glob.glob(source_str)
Data_str=xr.open_mfdataset(files_str,combine="nested",concat_dim="scanID")
w_str=xr.DataArray(Data_str.wind_speed.where(Data_str.qc_wind_speed==0).values.T.squeeze(),coords={'time':Data_str.time.values.squeeze(),'range':Data_str.range.values}) 
w_str=w_str.resample(time='10s').nearest(tolerance='1s') 
w_str=interp_nan(mad_despike(w_str, window, max_mad_ws),limit_uvw)

#%% Main

for s in sites:
    Data_trp_avg=Data_trp[s].where(Data_trp[s].time<=Data_trp[s].time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')
    Data_trp[s]['dtheta']=Data_trp[s].theta-Data_trp_avg.theta

#%% Plots

#theta-r-p
s=site_sel
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1],width_ratios=[1,0.025])
ax=fig.add_subplot(gs[0,0])
cf=plt.contourf(Data_trp[s].time,Data_trp[s].height,Data_trp[s].theta.T,np.arange(T_min,T_max+.1,.5),cmap='hot',extend='both')
plt.contour(Data_trp[s].time,Data_trp[s].height,Data_trp[s].theta.T,np.arange(T_min,T_max+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
cax=fig.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [K]')

ax=fig.add_subplot(gs[1,0])
cf=plt.contourf(Data_trp[s].time,Data_trp[s].height,Data_trp[s].waterVapor.T,np.arange(r_min,r_max+.1,.5),cmap='GnBu',extend='both')
plt.contour(Data_trp[s].time,Data_trp[s].height,Data_trp[s].waterVapor.T,np.arange(r_min,r_max+.1,.5),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

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
plt.xlim([Data_trp[s].time[0],Data_trp[s].time[-1]])
plt.grid()
plt.tight_layout()

#dtheta
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(len(sites), 2, height_ratios=len(sites)*[1],width_ratios=[1,0.025])
ctr=0
for s in sites:
    ax=fig.add_subplot(gs[ctr,0])
    cf=plt.contourf(Data_trp[s].time,Data_trp[s].height,Data_trp[s].dtheta.T,np.arange(-5,5+.1,0.25),cmap='seismic',extend='both')
    plt.contour(Data_trp[s].time,Data_trp[s].height,Data_trp[s].dtheta.T,np.arange(-5,5+.1,0.25),colors='k',linewidths=1,alpha=0.25,extend='both')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel(r'$z$ [m]')
    plt.grid()
    plt.ylim([0,300])
    if ctr<len(sites)-1:
        ax.set_xticklabels([])
    ctr+=1
plt.xlabel('Time (UTC)')
cax=fig.add_subplot(gs[:,1])
plt.colorbar(cf,cax=cax,label=r'$\theta^\prime$ [K]')
plt.tight_layout()

#wind maps
fig=plt.figure(figsize=(18,7))
gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1],width_ratios=[1,0.025])
ax=fig.add_subplot(gs[0,0])
cf=plt.contourf(Data_uvw.time,Data_uvw.height,Data_uvw.ws.T,np.arange(2,22+.1),cmap='coolwarm',extend='both')
plt.contour(Data_uvw.time,Data_uvw.height,Data_uvw.ws.T,np.arange(2,22+.1),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([Data_trp[s].time[0],Data_trp[s].time[-1]])
cax=fig.add_subplot(gs[0,1])
plt.colorbar(cf,cax=cax,label=r'$U$ [m s$^{-1}$]')

ax=fig.add_subplot(gs[1,0])
cf=plt.contourf(Data_uvw.time,Data_uvw.height,Data_uvw.wd.T,np.arange(45,336,10),cmap='gist_stern',extend='both')
plt.contour(Data_uvw.time,Data_uvw.height,Data_uvw.wd.T,np.arange(45,336,10),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
ax.set_xticklabels([])
plt.xlim([Data_trp[s].time[0],Data_trp[s].time[-1]])
cax=fig.add_subplot(gs[1,1])
plt.colorbar(cf,cax=cax,label=r'$\theta$ [$^\circ$]')

ax=fig.add_subplot(gs[2,0])
cf=plt.contourf(Data_uvw.time,Data_uvw.height,Data_uvw.w.T,np.arange(-5,5.1,0.25),cmap='seismic',extend='both')
plt.contour(Data_uvw.time,Data_uvw.height,Data_uvw.w.T,np.arange(-5,5.1,0.25),colors='k',linewidths=1,alpha=0.25,extend='both')
cf=plt.contourf(w_str.time,w_str.range,w_str.T,np.arange(-5,5.1,0.25),cmap='seismic',extend='both')
plt.contour(w_str.time,w_str.range,w_str.T,np.arange(-5,5.1,0.25),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.ylim([0,2000])
plt.xlabel('Time (UTC)')
plt.xlim([Data_trp[s].time[0],Data_trp[s].time[-1]])
cax=fig.add_subplot(gs[2,1])
plt.colorbar(cf,cax=cax,label=r'$w$ [m s$^{-1}$]')
plt.tight_layout()
