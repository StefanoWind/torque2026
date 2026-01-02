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
import matplotlib.dates as mdates
import matplotlib.cm as cm
import glob
import matplotlib.gridspec as gridspec
import warnings
import utm
import pyart
import pandas as pd
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.00')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
wf='King Plains'
D=127
t1=2000
t2=4000
skip=300#[s]
power_rated=2800
z_rad=100
make_video=False
time_shift_rad=120#[s] shift radar time

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
files_rad=glob.glob(source_rad)
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
Data=xr.Dataset()
os.makedirs(os.path.join(cd,'figures','yaw'),exist_ok=True)

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
                # scada_df=scada_df.rename(columns={c: c.split('Turbine')[-1]})
                turbines=np.append(turbines,c.split('Turbine')[-1][:2])
                _vars=np.append(_vars,c.split('.')[-1])
    turbines=np.unique(turbines)
    _vars=np.unique(_vars)
    
    #build dataset
    scada_ds=xr.Dataset()
    for v in _vars:
        data=np.zeros((len(scada_df.index),len(turbines)))
        i_t=0
        for t in turbines:
            data[:,i_t]=scada_df[f'PKGP1HIST01.OKWF001_KP_Turbine{t}.{v}'].values
            i_t+=1
        scada_ds[v]=xr.DataArray(data,coords={'time':scada_df.index.values,'turbine':turbines})
    
    #stack
    if 'time' not in Data.coords:
        Data=scada_ds
    else:
        Data=xr.concat([Data,scada_ds],dim='time')
        
    ctr+=1
    


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
    yaw_int=np.degrees(np.arctan2(s_int,c_int))
    
    #plot
    pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Grays',vmin=-10,vmax=50)
    
    for tid in Data.turbine.values:
        x_tur=Turbines.x_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000
        y_tur=Turbines.y_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000
        yaw=yaw_int.sel(turbine=tid).values
        power=power_int.sel(turbine=tid).values
        color=[int(c*255) for c in cmap(power/power_rated)[:-1]]
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
plt.colorbar(sc,cax=cax,label='Normalized power')
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
        yaw_int=np.degrees(np.arctan2(s_int,c_int))
        
        #plot
        pc=plt.pcolor((grid.x['data']+xy[0])/1000,(grid.y['data']+xy[1])/1000,Z,cmap='Grays',vmin=-10,vmax=50)
        
        for tid in Data.turbine.values:
            x_tur=Turbines.x_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000
            y_tur=Turbines.y_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000
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

