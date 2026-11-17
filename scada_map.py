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
source=os.path.join(cd,'data','awaken','kp.turbine.z02.b0','*20230805*nc')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
source_offset=os.path.join(cd,'data','wd_offsets_Sept2025.csv')

t_start='2023-08-05T10:00:00'
t_end='2023-08-05T11:59:59'
limit_time=10#[s] max gap filled by interpolation
D=127#[m] rotor diameter
power_rated=2820#[kW] rated power
max_range_rad=100#[km] max radar range read
levels_rad=np.arange(-10,51,5)#[dBZ] reflectivity contour levels

make_video=False

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

def read_lowest_tilt(file):
    radar=pyart.io.read_nexrad_archive(file).extract_sweeps([0])

    #filter bad data
    gatefilter=pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked("reflectivity")

    #gate-center locations along the curved beam, mapped to UTM through lat/lon
    n_gates=np.sum(radar.range['data']<=max_range_rad*1000)
    x,y,_=radar.get_gate_x_y_z(0)
    lon,lat=pyart.core.cartesian_to_geographic_aeqd(x[:,:n_gates],y[:,:n_gates],
                                                    radar.longitude['data'][0],radar.latitude['data'][0])
    x_utm,y_utm,_,_=utm.from_latlon(lat,lon,force_zone_number=14)

    Z=np.ma.masked_where(gatefilter.gate_excluded,radar.fields['reflectivity']['data'])[:,:n_gates]
    #mid-sweep time of the lowest tilt
    time_rad=np.datetime64(radar.time['units'][-20:-1])+np.timedelta64(int(np.round(np.mean(radar.time['data']))),'s')
    return x_utm/1000,y_utm/1000,Z,time_rad

#%% Initialization
files_rad=sorted(glob.glob(source_rad))
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
Data_offset=pd.read_csv(source_offset).set_index('Turbine Serial Number')['Northing Bias - 2022']
os.makedirs(os.path.join(cd,'figures','yaw'),exist_ok=True)

#read scada
scada_all=[]
for file in sorted(glob.glob(source)):
    scada_ds=xr.open_dataset(file).rename({'WTUR.DateTime':'time'})
    turbine=scada_ds.attrs['Alias']
    serial=int(scada_ds['WTUR.Id'][0])
    scada_ds=scada_ds.sel(time=slice(t_start,t_end))

    #yaw as unit vector to avoid averaging/interpolating across 0-360
    scada_ds=xr.Dataset({'ActivePower':scada_ds['WTUR.W'],
                         'cos_yaw':np.cos(np.radians(scada_ds['WNAC.Dir'])),
                         'sin_yaw':np.sin(np.radians(scada_ds['WNAC.Dir']))})

    #irregular (1-7 s) logging interval: bin to a common 1-s grid across turbines and fill short gaps
    scada_ds=scada_ds.resample(time='1s').mean().interpolate_na(dim='time',max_gap=np.timedelta64(limit_time,'s'))
    scada_ds['yaw_offset']=Data_offset[serial]
    scada_all.append(scada_ds.expand_dims(turbine=[turbine]))
Data=xr.concat(scada_all,dim='turbine')

#%% Plots
cmap = cm.get_cmap("plasma")
ctr=0
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(2, 3,height_ratios=[1,1],width_ratios=[1,1,0.05])
for f in np.array(files_rad)[sel_plot]:
    ax=fig.add_subplot(gs[int(ctr/2),np.mod(ctr,2)])
    x_gate,y_gate,Z,time_rad=read_lowest_tilt(f)

    #interpolate data
    power_int=Data.ActivePower.interp(time=time_rad)
    c_int=Data.cos_yaw.interp(time=time_rad)
    s_int=Data.sin_yaw.interp(time=time_rad)
    yaw_int=np.degrees(np.arctan2(s_int,c_int))+Data.yaw_offset

    #plot
    pc=plt.contourf(x_gate,y_gate,Z,levels=levels_rad,cmap='Grays',extend='both')
    plt.contour(x_gate,y_gate,Z,levels=levels_rad,colors='k',linewidths=0.3,alpha=0.5)

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
        x_gate,y_gate,Z,time_rad=read_lowest_tilt(f)

        #interpolate pressure data
        power_int=Data.ActivePower.interp(time=time_rad)
        c_int=Data.cos_yaw.interp(time=time_rad)
        s_int=Data.sin_yaw.interp(time=time_rad)
        yaw_int=np.degrees(np.arctan2(s_int,c_int))+Data.yaw_offset
        
        #plot
        pc=plt.contourf(x_gate,y_gate,Z,levels=levels_rad,cmap='Grays',extend='both')
        plt.contour(x_gate,y_gate,Z,levels=levels_rad,colors='k',linewidths=0.3,alpha=0.5)
        
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

