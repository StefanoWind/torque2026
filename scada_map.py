# -*- coding: utf-8 -*-
"""
Analyze SCADA data during bore
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import glob
import matplotlib.gridspec as gridspec
import warnings
import utm
import pandas as pd
import pyart
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','20230805.100000.20230805.115959.scada.nc')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
source_offset=os.path.join(cd,'data','wd_offsets_Sept2025.csv')

wf='King Plains'
D=127#[m] rotor diameter
t1=2000#initial timestep
t2=4000#final timestep
skip=300#skip timesteps
power_rated=2800#[kW] rated power
z_rad=100#[m] radar plane height a.g.l.
time_shift_rad=120#[s] shift radar time

make_video=True

#graphics
sel_plot=[0,8,11,16]

#%% Functions
def change_color(turbine_file,color):
    from PIL import Image
    img = Image.open(turbine_file).convert("RGBA")

    pixels = img.load()
    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            # Detect black (or near-black)
            if a > 0 and r < 50 and g < 50 and b < 50:
                pixels[x, y] = (*color, a)
    return img            
    
def draw_turbine(x,y,D,wd,turbine_file,color=(255, 0, 0)):
    from matplotlib import transforms
    from matplotlib import pyplot as plt
    
    img=change_color(turbine_file,color)
    
    ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    tr = transforms.Affine2D().scale(D/1800,D/1800).translate(-20*D/700,-350*D/700).rotate_deg(90-wd).translate(x,y)
    ax.imshow(img, transform=tr + ax.transData,zorder=10)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
#%% Initialization
Data=xr.open_dataset(source)
files_rad=glob.glob(source_rad)
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
Data_offset=pd.read_csv(source_offset)
os.makedirs(os.path.join(cd,'figures','yaw'),exist_ok=True)

#%% Plots
cmap = cm.get_cmap("plasma")
ctr=0
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 3,height_ratios=[1,1],width_ratios=[1,1,0.05])
for f in np.array(files_rad)[sel_plot]:
    ax=fig.add_subplot(gs[int(ctr/2),np.mod(ctr,2)])
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
        grid_limits=((z_rad, z_rad),  (-100000.0, 0.0),(50000, 80000.0),),
        fields=["reflectivity"],
    )
    
    #time-space info
    time_rad=np.timedelta64(time_shift_rad,'s')+np.datetime64(grid.time['units'][-20:])
    Z=grid.fields['reflectivity']['data'].squeeze()
    xy=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
    
    #interpolate pressure data
    power_int=Data.ActivePower.interp(time=time_rad)
    c_int=np.cos(np.radians(Data.Nacelle_Position.interp(time=time_rad)))
    s_int=np.sin(np.radians(Data.Nacelle_Position.interp(time=time_rad)))
    yaw_int=np.degrees(np.arctan2(s_int,c_int))+Data_offset['Northing Bias - 2022'].values
    
    #plot
    pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Grays',vmin=-10,vmax=50)
    
    for tid in Data.turbine.values:
        x_tur=Turbines.x_utm.where(Turbines.name==tid,drop=True).values/1000
        y_tur=Turbines.y_utm.where(Turbines.name==tid,drop=True).values/1000
        yaw=yaw_int.sel(turbine=tid).values
        power=power_int.sel(turbine=tid).values
        if power/power_rated>0.01:
            color=[int(c*255) for c in cmap(power/power_rated)[:-1]]
        else:
            color=[250,0,0]
        draw_turbine(x_tur, y_tur, D/100, yaw, os.path.join(cd,'figures','Turbine.png'),color)
   
    plt.gca().set_aspect('equal')
    
    if ctr>=2:
        plt.xlabel('W-E [km]')
    else:
        ax.set_xticklabels([])
    if ctr==0 or ctr==2:
        plt.ylabel('S-N [km]')
    else:
        ax.set_yticklabels([])
    plt.xlim([628,649])
    plt.ylim([4023,4034])
    plt.xticks(np.arange(630,646,5))
    plt.yticks(np.arange(4025,4031,5))
    plt.grid()
    plt.title(str(time_rad).replace('T',' '))
    ctr+=1
    
sc=ax.scatter(0,0,1,c=0,cmap='plasma',vmin=0,vmax=1)
cax=fig.add_subplot(gs[:,-1])
plt.colorbar(sc,cax=cax,label=r'$P_{norm}$')
plt.tight_layout()

if make_video:   
    ctr=0
    for f in files_rad:
        plt.figure(figsize=(16,6))
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
            grid_limits=((z_rad, z_rad),  (-100000.0, 0.0),(50000, 80000.0),),
            fields=["reflectivity"],
        )
        
        #time-space info
        time_rad=np.timedelta64(time_shift_rad,'s')+np.datetime64(grid.time['units'][-20:])
        Z=grid.fields['reflectivity']['data'].squeeze()
        xy=utm.from_latlon(radar.latitude['data'],radar.longitude['data'])
        
        #interpolate pressure data
        power_int=Data.ActivePower.interp(time=time_rad)
        c_int=np.cos(np.radians(Data.Nacelle_Position.interp(time=time_rad)))
        s_int=np.sin(np.radians(Data.Nacelle_Position.interp(time=time_rad)))
        yaw_int=np.degrees(np.arctan2(s_int,c_int))+Data_offset['Northing Bias - 2022'].values
        
        #plot
        pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Grays',vmin=-10,vmax=50)
        
        for tid in Data.turbine.values:
            x_tur=Turbines.x_utm.where(Turbines.name==tid,drop=True).values/1000
            y_tur=Turbines.y_utm.where(Turbines.name==tid,drop=True).values/1000
            yaw=yaw_int.sel(turbine=tid).values
            power=power_int.sel(turbine=tid).values
            color=[int(c*255) for c in cmap(power/power_rated)[:-1]]
            draw_turbine(x_tur, y_tur, D/100, yaw, os.path.join(cd,'figures','Turbine.png'),color)
       
        plt.gca().set_aspect('equal')
       
        plt.xlabel('W-E [km]')
        plt.ylabel('S-N [km]')
        plt.xlim([628,649])
        plt.ylim([4023,4034])
        plt.xticks(np.arange(630,646,5))
        plt.yticks(np.arange(4025,4031,5))
        plt.grid()
        sc=plt.scatter(0,0,1,c=0,cmap='plasma',vmin=0,vmax=1)
        plt.colorbar(sc,label='Normalized power')
        plt.tight_layout()
        plt.title(str(time_rad).replace('T',' '))
        plt.savefig(os.path.join(cd,'figures','yaw',f'{ctr:02.0f}'))
        plt.close()
        ctr+=1

