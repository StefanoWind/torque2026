# -*- coding: utf-8 -*-
"""
Plot data during a frontal passage
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pyart
import glob
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import warnings
import pandas as pd
from asammdf import MDF
import utm
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_msn=os.path.join(cd,'data','mesonet','*mdf')
source_msn_layout=os.path.join(cd,'data','mesonet','geoinfo.csv')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
sites_trp=['B','C1a','G']
wf_sel=['King Plains','Armadillo Flats','Breckinridge']

avg_time=10#
max_ele_rad=8
z_rad=100
time_shift_rad=0

#%% Initialiation
msn_lyt=pd.read_csv(source_msn_layout).set_index('stid')

xy=utm.from_latlon(msn_lyt['nlat'].values, msn_lyt['elon'].values)
msn_lyt['x']=xy[0]
msn_lyt['y']=xy[1]

files_msn=glob.glob(source_msn)

files_rad=glob.glob(source_rad)

Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

os.makedirs(os.path.join(cd,'figures','mesonet'),exist_ok=True)

#%% Main
Data_msn=xr.Dataset()
for f in files_msn:
    df = pd.read_csv(f,delim_whitespace=True, skiprows=2,na_values=[-998, -999])
    datestr=str(os.path.basename(f))[:-4]
    time=np.datetime64(f'{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[8:10]}:{datestr[10:12]}:00')
    
    ds = xr.Dataset.from_dataframe(df.set_index("STID"))
    ds = ds.expand_dims(time=[time])
    
    if 'PRES' in Data_msn.data_vars:
        Data_msn=xr.concat([Data_msn, ds],dim='time')
    else:
        Data_msn=ds
        
dp=(Data_msn.PRES/Data_msn.PRES.where(Data_msn.time<Data_msn.time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')-1)*100

x_msn=[]
y_msn=[]
for s in Data_msn.STID.values:
    x_msn=np.append(x_msn,msn_lyt['x'].loc[s])
    y_msn=np.append(y_msn,msn_lyt['y'].loc[s])
    
#AWAKEN turbines
Turbines['wind_plant']=Turbines.wind_plant.where(Turbines.wind_plant!='unknown Garfield County','Armadillo Flats')
x_tur=[]
y_tur=[]
for wf in wf_sel:
    x_tur=np.append(x_tur,Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values)
    y_tur=np.append(y_tur,Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values)
    
#NEXRAD
ctr=0
for f in files_rad:
    plt.figure()
    radar = pyart.io.read_nexrad_archive(f)
    
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked("reflectivity")
    
    grid = pyart.map.grid_from_radars(
        (radar,),
        gatefilters=(gatefilter,),
        grid_shape=(1, 241, 241),
        grid_limits=((z_rad, z_rad),  (-250000.0, 60000.0),(-105000, 200000.0),),
        fields=["reflectivity"],
    )
    
    time_rad=np.timedelta64(time_shift_rad,'s')+np.datetime64(grid.time['units'][-20:])
    Z=grid.fields['reflectivity']['data'].squeeze()
    xy=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
    
    #interpolate pressure data
    dp_int=dp.interp(time=time_rad)
    
    pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Spectral_r')
    plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
    sc=plt.scatter(x_msn/1000,y_msn/1000,s=30,c=dp_int,cmap='seismic',vmin=-0.5,vmax=0.5,edgecolor='k')
    
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('W-E [km]')
    plt.ylabel('S-N [km]')
    plt.xlim([475,700])
    plt.ylim([3925,4125])
    plt.grid()
    plt.title(str(time_rad).replace('T',' ')+' UTC')
    
    plt.colorbar(pc,label=r'$Z$ [dBZ]',location='right')
    plt.colorbar(sc,label=r'$\Delta p$ [%]',location='right')
    
    plt.savefig(os.path.join(cd,'figures','mesonet',f'{ctr:02.0f}_mesonet.png'))
    plt.close()
    ctr+=1
    
#%% Plots

for it in range(len(Data_msn.time)):
    
    plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
    sc=plt.scatter(x_msn/1000,y_msn/1000,s=20,c=dp.isel(time=it),cmap='seismic',vmin=-1,vmax=1,edgecolor='k')
    plt.title(str(Data_msn.time.values[it]).replace('T',' '))
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('W-E [km]')
    plt.ylabel('S-N [km]')
    plt.grid()
    plt.colorbar(sc,label=r'$\Delta p$ [%]')
    plt.xlim([x_tur.min()/1000-200,x_tur.max()/1000+75])
    plt.ylim([y_tur.min()/1000-200,y_tur.max()/1000+75])

    plt.savefig(os.path.join(cd,'figures','mesonet',f'{it:02.0f}_mesonet.png'))
    plt.close()