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
import pyart
import glob
import matplotlib.gridspec as gridspec
import warnings
import pandas as pd
import utm
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_msn=os.path.join(cd,'data','mesonet','*mdf')
source_msn_layout=os.path.join(cd,'data','mesonet','geoinfo.csv')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
sites_trp=['B','C1a','G']
wf_sel=['King Plains','Armadillo Flats','Breckinridge']

avg_time=10#[s] prebore period
z_rad=100#[m a.g.l.] radar plane
time_shift_rad=120#[s] shift radar time

make_video=True

#graphics
sel_plot=[0,8,16]

#%% Initialiation

#read mesonet layout
msn_lyt=pd.read_csv(source_msn_layout).set_index('stid')
xy=utm.from_latlon(msn_lyt['nlat'].values, msn_lyt['elon'].values)
msn_lyt['x']=xy[0]
msn_lyt['y']=xy[1]

#read AWAKEN leyout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#load data
files_msn=glob.glob(source_msn)
files_rad=glob.glob(source_rad)

os.makedirs(os.path.join(cd,'figures','mesonet'),exist_ok=True)

#%% Main

#read mesonet data
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

#pressure perturbation
dp=(Data_msn.PRES/Data_msn.PRES.where(Data_msn.time<Data_msn.time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')-1)*100

#reconcile station locations
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
    

#%% Plots

#plot selected frames
fig=plt.figure(figsize=(18,5))
gs = gridspec.GridSpec(1, 5,width_ratios=[1,1,1,0.05,0.05])

ctr=0
for f in np.array(files_rad)[sel_plot]:
    ax=fig.add_subplot(gs[0,ctr])

    radar = pyart.io.read_nexrad_archive(f)
    
    #filter bad data
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked("reflectivity")
    
    #extract Cartesian slice
    grid = pyart.map.grid_from_radars(
        (radar,),
        gatefilters=(gatefilter,),
        grid_shape=(1, 241, 241),
        grid_limits=((z_rad, z_rad),  (-250000.0, 60000.0),(-105000, 200000.0),),
        fields=["reflectivity"],
    )
    
    #time-space info
    time_rad=np.timedelta64(time_shift_rad,'s')+np.datetime64(grid.time['units'][-20:])
    Z=grid.fields['reflectivity']['data'].squeeze()
    xy=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
    
    #interpolate pressure data
    dp_int=dp.interp(time=time_rad)
    
    #plot
    pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Spectral_r',vmin=-10,vmax=50)
    plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
    sc=plt.scatter(x_msn/1000,y_msn/1000,s=50,c=dp_int,cmap='seismic',vmin=-0.5,vmax=0.5,edgecolor='k',linewidth=2)
    
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('W-E [km]')
    plt.xlabel('W-E [km]')
    if ctr==0:
        plt.ylabel('S-N [km]')
    else:
        ax.set_yticklabels([])
    plt.xlim([475,700])
    plt.ylim([3925,4125])
    plt.xticks(np.arange(475,701,25))
    plt.yticks(np.arange(3925,4126,25))
    plt.grid()
    plt.title(str(time_rad).replace('T',' ')+' UTC')
    ctr+=1
    
cax=fig.add_subplot(gs[0,-2])    
cbar=plt.colorbar(pc,cax=cax)
cbar.ax.set_title(r'$Z$ [dBZ]  ', pad=10)
cax=fig.add_subplot(gs[0,-1])    
cbar=plt.colorbar(sc,cax=cax)
cbar.ax.set_title(r'  $\Delta p$ [%]', pad=10)  
cbar.ax.set_yticks([-0.5,0,0.5])  

#plot all    
if make_video:   
    ctr=0
    for f in files_rad:
        plt.figure(figsize=(12,7))
        radar = pyart.io.read_nexrad_archive(f)
        
        #filter bad data
        gatefilter = pyart.filters.GateFilter(radar)
        gatefilter.exclude_transition()
        gatefilter.exclude_masked("reflectivity")
        
        #extract Cartesian slice
        grid = pyart.map.grid_from_radars(
            (radar,),
            gatefilters=(gatefilter,),
            grid_shape=(1, 241, 241),
            grid_limits=((z_rad, z_rad),  (-250000.0, 60000.0),(-105000, 200000.0),),
            fields=["reflectivity"],
        )
        
        #time-space info
        time_rad=np.timedelta64(time_shift_rad,'s')+np.datetime64(grid.time['units'][-20:])
        Z=grid.fields['reflectivity']['data'].squeeze()
        xy=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
        
        #interpolate pressure data
        dp_int=dp.interp(time=time_rad)
        
        #plot
        pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Spectral_r',vmin=-10,vmax=50)
        plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
        sc=plt.scatter(x_msn/1000,y_msn/1000,s=50,c=dp_int,cmap='seismic',vmin=-0.5,vmax=0.5,edgecolor='k',linewidth=2)
        
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
        plt.tight_layout()
        plt.savefig(os.path.join(cd,'figures','mesonet',f'{ctr:02.0f}_mesonet.png'))
        plt.close()
        ctr+=1
            
        