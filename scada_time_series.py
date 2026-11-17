# -*- coding: utf-8 -*-
"""
Plot selected SCADA channels
"""
import os
import glob
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.colors import to_rgb
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.b0')
t_start='2023-08-05T10:00:00'
t_end='2023-08-05T11:59:59'
var_names={'WMET.HorWdSpd':'WindSpeed','WTUR.W':'ActivePower','WNAC.Dir':'Nacelle_Position'}
limit_time=10#[s] max gap filled by interpolation
turbine_id={'A09':'wt009','I02':'wt073','G09':'wt062'}
turbines_sel=['A09','I02','G09']
U_cutin=3#[m/s] cutin wind speed
U_rated=11#[m/s] rated wind speed
U_cutoff=25#[m/s] #cut-off wind speed
vars_sel2=['WindSpeed','ActivePower','Nacelle_Position']
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
source_offset=os.path.join(cd,'data','wd_offsets_Sept2025.csv')
source_icon=os.path.join(cd,'figures','Turbine.png')
dt_ma=60#[s] moving average window
dt_yaw=60#[s] sampling interval of the yaw icons

#graphics
colors={'A09':'g','I02':'b','G09':'r'}
labels={'WindSpeed':r'$U_h$ [m s$^{-1}$]','ActivePower':r'$P_{norm}$','Nacelle_Position':''}
norm={'WindSpeed':1,'ActivePower':2820,'Nacelle_Position':1}
D_icon=2.5#[min] rotor size of the yaw icons

#%% Functions
def change_color(img,color):
    pixels=np.array(img)
    black=(pixels[...,3]>0)&np.all(pixels[...,:3]<50,axis=-1)
    pixels[black,:3]=color
    return Image.fromarray(pixels)

def draw_turbine(x,y,D,wd,img):
    ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    tr = transforms.Affine2D().scale(D/1800,D/1800).translate(-20*D/700,-350*D/700).rotate_deg(90-wd).translate(x,y)
    ax.imshow(img, transform=tr + ax.transData,zorder=10,aspect='auto')
    plt.xlim(xlim)
    plt.ylim(ylim)

#%% Main
Data_offset=pd.read_csv(source_offset).set_index('Turbine Serial Number')['Northing Bias - 2022']

#read scada
scada_all=[]
for tid in turbines_sel:
    files=glob.glob(os.path.join(source,f'*{t_start[:10].replace("-","")}*'+turbine_id[tid]+'*'))
    scada_ds=xr.open_mfdataset(files).rename({'WTUR.DateTime':'time'})
    serial=int(scada_ds['WTUR.Id'][0])
    scada_ds=scada_ds[list(var_names)].rename(var_names).sel(time=slice(t_start,t_end)).compute()

    #irregular (1-7 s) logging interval: bin to a common 1-s grid across turbines and fill short gaps
    scada_ds=scada_ds.resample(time='1s').mean().interpolate_na(dim='time',max_gap=np.timedelta64(limit_time,'s'))
    scada_ds['yaw_offset']=Data_offset[serial]
    scada_all.append(scada_ds.expand_dims(turbine=[tid]))

Data=xr.concat(scada_all,dim='turbine',combine_attrs='drop')
dt=np.float64(np.mean(np.diff(Data.time.values)))/10**9

#yaw downsampled to 1 min as unit vector to avoid averaging across 0-360, bias-corrected and sampled every dt_yaw
cos_yaw=np.cos(np.radians(Data.Nacelle_Position)).resample(time='1min').mean()
sin_yaw=np.sin(np.radians(Data.Nacelle_Position)).resample(time='1min').mean()
yaw_1min=(np.degrees(np.arctan2(sin_yaw,cos_yaw))+Data.yaw_offset)%360
yaw_sel=yaw_1min.sel(time=np.arange(yaw_1min.time.values[0],yaw_1min.time.values[-1],np.timedelta64(dt_yaw,'s')))
icons={t:change_color(Image.open(source_icon).convert('RGBA'),[int(c*255) for c in to_rgb(colors[t])]) for t in turbines_sel}

#%% Plots
plt.close('all')


ctr=1
plt.figure(figsize=(18,7))
for v in vars_sel2:
    ax=plt.subplot(len(vars_sel2),1,ctr)
    if ctr==1:
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutin,U_cutin],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_rated,U_rated],'--k')
        plt.plot([Data.time.values[0],Data.time.values[-1]],[U_cutoff,U_cutoff],'--k')
    plt.xlim([Data.time[0],Data.time[-1]+np.timedelta64(1,'s')])
    if v=='Nacelle_Position':
        #y range matched to the panel's physical aspect ratio so that the icons are not distorted
        pos=ax.get_position()
        fig_w,fig_h=plt.gcf().get_size_inches()
        x_range=np.diff(ax.get_xlim())[0]
        y_range=x_range*pos.height*fig_h/(pos.width*fig_w)
        plt.ylim([-y_range/2,y_range/2])
        y_icon={t:(len(turbines_sel)-1-2*i)*y_range/(2*len(turbines_sel)) for i,t in enumerate(turbines_sel)}
        for t in turbines_sel:
            for tm in yaw_sel.time.values:
                yaw=float(yaw_sel.sel(turbine=t,time=tm))
                if ~np.isnan(yaw):
                    draw_turbine(mdates.date2num(tm),y_icon[t],D_icon/1440,yaw,icons[t])
                
        for tl,t in zip(ax.get_yticklabels(),y_icon):
            tl.set_color(colors[t])
        ax.grid(axis='x')
        ax.set_yticks([])
    else:
        for t in turbines_sel:
            plt.plot(Data.time,Data[v].sel(turbine=t)/norm[v],alpha=0.25,color=colors[t],linewidth=1)
            plt.plot(Data.time,Data[v].rolling(time=int(dt_ma/dt),center=True).mean().sel(turbine=t)/norm[v],color=colors[t])
        plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if ctr<len(vars_sel2):
        ax.set_xticklabels([])
    plt.ylabel(labels[v])
    plt.xlim([Data.time[0]-np.timedelta64(10,'s'),Data.time[-1]+np.timedelta64(10,'s')])
    ctr+=1
plt.xlabel('Time (UTC)')



    
    