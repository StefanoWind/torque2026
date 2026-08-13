# -*- coding: utf-8 -*-
"""
Plot met station time series from Highmore (HIHS2)
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from nexradaws import NexradAwsInterface
from datetime import timedelta, datetime,timezone
import warnings
from pathlib import Path
import pyart
import cartopy.crs as ccrs
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source=os.path.join(cd,'data','HIHS2.csv')
source_lt=os.path.join(cd,'data','SDMesonet_HIHS2_2021_2026_FMO.csv')
source_turbines=os.path.join(cd,'data','turbines_SD.xlsx')

#time series
arrow_skip=5#plot a direction arrow every N samples
arrow_height_frac=1.15#height of direction arrows above max(US,UG)

#data selection
sdate=np.datetime64('2026-06-29 00:00:00')
edate=np.datetime64('2026-06-30 00:00:00')
sdate_narrow=np.datetime64('2026-06-29 10:30:00')
edate_narrow=np.datetime64('2026-06-29 12:30:00')

#radar video
make_radar_video=True
radar_id_narrow='KABR'
min_lon_narrow,max_lon_narrow=-101,-98
min_lat_narrow,max_lat_narrow=43.5,45.5
lat_met=44.525775
lon_met=-99.461363

#histograms
sel_hist=['US',
          'UE',
          'UG']

var_label={'TA':r'$T$ [$^\circ$C]','PP':'$PP$ [mm]','US':r'$U$ [m s$^{-1}$]',
           'UD':r'$\theta$ [$^\circ$]','UE':r'$\sigma_{\theta}$ [$^\circ$]',
           'UG':r'$U_{gust}$ [m s$^{-1}$]','UH':r'$\theta_{gust}$ [$^\circ$]',
           'XR':'$RH$ [%]','RW':r'$RW$ [W m$^{-2}$]','PA':'$p$ [mbar]'}

#2D pdf
ug_bin_width=1#[m/s] wind gust bin width for the 2D occurrence histogram
ue_bin_width=2#[deg] wind direction std bin width for the 2D occurrence histogram

#%% Initialization
Data=pd.read_csv(source,skiprows=2)
Data['TmStamp']=pd.to_datetime(Data['TmStamp'],format='%d-%m-%y %H:%M')
Data=Data.set_index('TmStamp')

Data_lt=pd.read_csv(source_lt)
Data_lt.columns=[c.strip() for c in Data_lt.columns]
Data_lt['TmStamp']=pd.to_datetime(Data_lt['Date']+' '+Data_lt['Time (UTC)'])
Data_lt=Data_lt.set_index('TmStamp').drop(columns=['Date','Time (UTC)'])

#rename to match HIHS2.csv variable codes (units already consistent: C, mm, m/s, deg, %, W/m^2, mbar)
Data_lt=Data_lt.rename(columns={'Temperature (C)':'TA',
                                 'Precipitation (mm)':'PP',
                                 'Wind Speed (m/s)':'US',
                                 'Wind Direction (deg)':'UD',
                                 'Wind Direction Standard Deviation (deg)':'UE',
                                 'Wind Gust Speed (m/s)':'UG',
                                 'Wind Gust Direction (deg)':'UH',
                                 'Relative Humidity (%)':'XR',
                                 'Solar Radiation (w/m2)':'RW',
                                 'Pressure (mbar)':'PA'})

date_form=mdates.ConciseDateFormatter(mdates.AutoDateLocator())

#%% Radar video (sequence of frames spanning sdate_narrow-edate_narrow, with turbines overlaid)
if make_radar_video:
    Turbines=pd.read_excel(source_turbines)

    video_dir=os.path.join(cd,'figures','radar_video')
    os.makedirs(video_dir,exist_ok=True)

    conn=NexradAwsInterface()
    scans=conn.get_avail_scans_in_range(pd.Timestamp(sdate_narrow).to_pydatetime().replace(tzinfo=timezone.utc),
                                         pd.Timestamp(edate_narrow).to_pydatetime().replace(tzinfo=timezone.utc),
                                         radar_id_narrow)
    scans=[s for s in scans if not s.filename.endswith('_MDM')]#drop metadata-only companion files (not valid Level II volumes)
    scans=sorted(scans,key=lambda s: s.scan_time)
    downloaded=conn.download(scans,os.path.join(cd,'data'))

    for ctr,f in enumerate(downloaded.success):
        radar=pyart.io.read_nexrad_archive(f.filepath)
        time_rad=pyart.util.datetime_from_radar(radar)

        with plt.style.context("dark_background"):
            display=pyart.graph.RadarMapDisplay(radar)
            fig=plt.figure(figsize=(10,8),facecolor="black")
            display.plot_ppi_map("reflectivity",sweep=0,
                                  min_lon=min_lon_narrow,max_lon=max_lon_narrow,
                                  min_lat=min_lat_narrow,max_lat=max_lat_narrow,
                                  resolution="10m",vmin=-10,vmax=70)

            ax=plt.gca()
            ax.scatter(Turbines.xlong,Turbines.ylat,s=3,color='w',transform=ccrs.PlateCarree(),zorder=6)
            display.plot_point(lon_met,lat_met,symbol='wo',markersize=8)
            display.plot_point(lon_met,lat_met,symbol='k+',markersize=7)

            plt.title(time_rad.strftime('%Y-%m-%d %H:%M:%S'),color='white')
            plt.savefig(os.path.join(video_dir,f'{ctr:03.0f}.png'),
                        dpi=300,bbox_inches="tight",facecolor=fig.get_facecolor())
        plt.close()

#%% Plots
fig=plt.figure(figsize=(18,9))
gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])

#pressure
ax=fig.add_subplot(gs[0,0])
plt.plot(Data.index,Data.PA,'k')
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel('$p$ [hPa]')
plt.grid()
ax.set_xticklabels([])
plt.xlim([Data.index[0],Data.index[-1]])

#wind speed, gust
ax=fig.add_subplot(gs[1,0])
plt.plot(Data.index,Data.US,'k',label=r'$U$')
plt.plot(Data.index,Data.UG,color=(1,0,0),label=r'$U_{gust}$')
time_arrow=Data.index[::arrow_skip]
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel(r'$U$ [m s$^{-1}$]')
plt.legend()
plt.grid()
ax.set_xticklabels([])
plt.xlim([sdate,edate])

#wind speed, gust and direction
ax=fig.add_subplot(gs[2,0])
plt.plot(Data.index,Data.UE,'k')
time_arrow=Data.index[::arrow_skip]
ud_arrow=Data.UD.values[::arrow_skip]
plt.quiver(time_arrow,np.ones(len(time_arrow))*50,
           np.cos(np.radians(270-ud_arrow)),np.sin(np.radians(270-ud_arrow)),
           color='r',width=0.002,scale=50)
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel(r'$\sigma_\theta$ [$^\circ$]')
plt.grid()
ax.set_xticklabels([])
plt.xlim([sdate,edate])

#temperature
ax=fig.add_subplot(gs[3,0])
plt.plot(Data.index,Data.TA,'k')
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel(r'$T$ [$^\circ$C]')
plt.grid()
plt.xlim([sdate,edate])
plt.xlabel('Time (UTC)')

plt.tight_layout()

#%% Long-term climatology histograms

Data_sel=Data.loc[sdate:edate]

variables=Data_lt.columns
fig=plt.figure(figsize=(18,5))
gs = gridspec.GridSpec(1,len(sel_hist))
for i,v in enumerate(sel_hist):
    ax=fig.add_subplot(gs[i//5,i%5])
    bins=np.linspace(0,np.nanmax(Data_lt[v]),50)
    H,xedges=np.histogram(Data_lt[v].dropna(),bins=bins)
    xc=(xedges[:-1]+xedges[1:])/2
    plt.fill_between(xc,H/H.sum(),xc*0,color='k',alpha=0.8)
    x_max=Data_sel.max()[v]
    p_exceed=(Data_lt[v].dropna()>x_max).mean()
    plt.plot([x_max,x_max],[0,0.25],'--r',lw=3)
    plt.xlabel(var_label[v])
    if i==0:
        plt.ylabel('Probability')
    plt.title(f'$P(X>{x_max:.1f})={p_exceed*100:.2f}\\%$')
    plt.grid()
plt.tight_layout()

#%% 2D probability of wind gust vs. wind direction std
Data_sel=Data.loc[sdate_narrow:edate_narrow]
ug_bins=np.arange(0,np.nanmax(Data_lt.UG)+ug_bin_width,ug_bin_width)
ue_bins=np.arange(0,np.nanmax(Data_lt.UE)+ue_bin_width,ue_bin_width)

valid_hist=np.isfinite(Data_lt.UG.values)&np.isfinite(Data_lt.UE.values)
H,xedges,yedges=np.histogram2d(Data_lt.UG.values[valid_hist],Data_lt.UE.values[valid_hist],bins=[ug_bins,ue_bins])
logP=np.ma.masked_invalid(np.log10(H.T/H.sum()))#log10 probability of occurrence, zero-count bins masked
xc=(xedges[:-1]+xedges[1:])/2
yc=(yedges[:-1]+yedges[1:])/2

pct_levels=np.array([0.0001,0.001,0.01,0.1,1,10])#probability levels, in percent
log_levels=np.log10(pct_levels/100)

fig=plt.figure(figsize=(8,6))
ax=plt.gca()
cf=ax.contourf(xc,yc,logP,levels=log_levels,cmap='Greys')
cbar=plt.colorbar(cf,ax=ax,label='Probability',pad=0.02)
cbar.set_ticks(log_levels)
cbar.set_ticklabels([f'{p:g}%' for p in pct_levels])

#selected period plotted on top as an oriented (arrow-connected) trajectory
ug_traj=Data_sel.UG.values
ue_traj=Data_sel.UE.values
for j in range(len(ug_traj)-1):
    if ~np.isnan(ug_traj[j]+ue_traj[j]+ug_traj[j+1]+ue_traj[j+1]):
        arrow=FancyArrowPatch((ug_traj[j],ue_traj[j]),(ug_traj[j+1],ue_traj[j+1]),
                               arrowstyle='->',color='r',mutation_scale=15,zorder=10)
        ax.add_patch(arrow)
plt.scatter(ug_traj,ue_traj,s=30,color='r',zorder=11,label='Selected period')
plt.xlabel(var_label['UG'])
plt.ylabel(var_label['UE'])
plt.legend()
plt.grid()
plt.tight_layout()
