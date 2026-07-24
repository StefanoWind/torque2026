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
from nexradaws import NexradAwsInterface
from datetime import timedelta, datetime,timezone
import warnings
from pathlib import Path
import pyart
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source=os.path.join(cd,'data','HIHS2.csv')
arrow_skip=5#plot a direction arrow every N samples
arrow_height_frac=1.15#height of direction arrows above max(US,UG)

sdate=np.datetime64('2026-06-29 00:00:00')
edate=np.datetime64('2026-06-30 00:00:00')

#%% Functions

def download_plot_nexrad(target_time: datetime,
                         radar_id: str = 'KVNX',
                         radar_dir: Path = 'data/',
                         min_lon: int = -101,
                         max_lon: int = -94,
                         min_lat: int = 33,
                         max_lat: int = 39,
                         vmin: int = -10,
                         vmax: int = 70,
                         site_lat: float = None,
                         site_lon: float = None,
                         site_label: str = None,
                         save_fig = False):

    # Find nearest scan
    conn = NexradAwsInterface()

    start = target_time - timedelta(minutes=10)
    end = target_time + timedelta(minutes=10)

    scans = conn.get_avail_scans_in_range(start, end, radar_id)

    if len(scans) == 0:
        raise ValueError(
            f"No scans found for {radar_id} between {start} and {end}"
        )

    nearest_scan = min(
        scans,
        key=lambda s: abs(s.scan_time - target_time)
    )

    # Download file
    downloaded_files = conn.download(
        [nearest_scan],
        str(radar_dir)
    )

    filename = downloaded_files.success[0].filepath

    # Read radar
    radar = pyart.io.read_nexrad_archive(filename)

    # Plot reflectivity (dark background scoped to this figure only, so it
    # doesn't leak into subsequently drawn time series / SHAP plots)
    with plt.style.context("dark_background"):
        display = pyart.graph.RadarMapDisplay(radar)

        fig = plt.figure(figsize=(10,8), facecolor="black")

        display.plot_ppi_map(
            "reflectivity",
            sweep=0,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            resolution="10m",
            vmin=vmin,
            vmax=vmax,
        )

        # Overlay the selected site location if coordinates are available.
        if site_lat is not None and site_lon is not None and np.isfinite(site_lat) and np.isfinite(site_lon):
            try:
                display.plot_point(site_lon, site_lat, symbol='wo', markersize=11)
                display.plot_point(site_lon, site_lat, symbol='k+', markersize=10)
            except Exception:
                ax = plt.gca()
                ax.plot(site_lon, site_lat, marker='o', markersize=8,
                        markerfacecolor='white', markeredgecolor='black', zorder=6)
                ax.plot(site_lon, site_lat, marker='+', markersize=9,
                        color='black', zorder=7)

            if site_label:
                try:
                    ax = plt.gca()
                    ax.text(site_lon + 0.06, site_lat + 0.04, site_label,
                            color='white', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.18', facecolor='black', alpha=0.5, linewidth=0),
                            zorder=8)
                except Exception:
                    pass

        if save_fig:
            radar_file = Path(filename)

            output_png = radar_file.with_suffix(".png")

            plt.savefig(
                output_png,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

    return fig

#%% Initialization
Data=pd.read_csv(source,skiprows=2)
Data['TmStamp']=pd.to_datetime(Data['TmStamp'],format='%d-%m-%y %H:%M')
Data=Data.set_index('TmStamp')

date_form=mdates.ConciseDateFormatter(mdates.AutoDateLocator())

download_plot_nexrad(target_time=datetime.strptime('2026-06-29 11:45','%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc),
                     radar_id='KABR',
                     min_lon= -101,
                     max_lon= -98,
                     min_lat= 43,
                     max_lat = 46,
                     vmin=-10,
                     vmax= 70,
                     site_lat=44.525775,
                     site_lon=-99.461363,
                     site_label='Highmore')
                     
                     

#%% Plots
fig=plt.figure(figsize=(18,9))
gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])

#pressure
ax=fig.add_subplot(gs[0,0])
plt.plot(Data.index,Data.PA,'k')
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel('$p$ [hPa]')
plt.grid()
ax.set_xticklabels([])
plt.xlim([Data.index[0],Data.index[-1]])

#wind speed, gust and direction
ax=fig.add_subplot(gs[1,0])
plt.plot(Data.index,Data.US,'k',label=r'$U$')
plt.plot(Data.index,Data.UG,color=(1,0,0),label=r'$U_{gust}$')
time_arrow=Data.index[::arrow_skip]
ud_arrow=Data.UD.values[::arrow_skip]
plt.quiver(time_arrow,Data.US[::arrow_skip]+1,
           np.cos(np.radians(270-ud_arrow)),np.sin(np.radians(270-ud_arrow)),
           color='k',width=0.002)
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel(r'$U$ [m s$^{-1}$]')
plt.legend()
plt.grid()
ax.set_xticklabels([])
plt.xlim([sdate,edate])

#temperature
ax=fig.add_subplot(gs[2,0])
plt.plot(Data.index,Data.TA,'k')
plt.gca().xaxis.set_major_formatter(date_form)
plt.ylabel(r'$T$ [$^\circ$C]')
plt.grid()
plt.xlim([sdate,edate])
plt.xlabel('Time (UTC)')

plt.tight_layout()
