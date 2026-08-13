# -*- coding: utf-8 -*-
"""
Scatterplot of turbine loads vs. wind speed with bin-averaged statistics
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
import xarray as xr
import glob
import warnings
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_loads=os.path.join(cd,'data/awaken/kp.turbine.z03.del/*nc')
source_scada=os.path.join(cd,'data/awaken/kp.turbine.z02.c0')
turbine_id='e6'
turbine_id_scada='wt041'#SCADA turbine id corresponding to turbine_id (Alias E06)
loads_var=['tb_bend_resultant','b1_bend_root_resultant']
ws_var='WMET.HorWdSpd_10m_Avg'
ws_flag_var='WS_Flag_10m_Percent'
pow_var='WTUR.W_10m_Avg'
pow_flag_var='PWR_Flag_10m_Percent'
min_power=2800*0.05
min_data_avail=0.5

ws_bin_width=1#[m/s] wind speed bin width
load_bin_width=0.05#[-] normalized load bin width for the 2D occurrence histogram
max_ci_frac=0.1#max width of the bootstrap CI, as a fraction of the load's std

highlight_period=[np.datetime64('2023-08-05T10:00:00'),np.datetime64('2023-08-05T12:00:00')]#datetimes highlighted in the scatter

labels={'tb_bend_resultant':'DEL of tower-base bending moment [kNm]',
        'b1_bend_root_resultant':'DEL of blade-root bending moment [kNm]'}

#%% Load data
files=sorted(glob.glob(source_loads))
Loads=xr.open_mfdataset(files).compute()

#restrict SCADA files to the days covered by the loads dataset
dates=pd.date_range(pd.Timestamp(Loads.time.values.min()).floor('D'),
                     pd.Timestamp(Loads.time.values.max()).floor('D'))
files_scada=[]
for d in dates:
    files_scada+=glob.glob(os.path.join(source_scada,f'*{d.strftime("%Y%m%d")}*{turbine_id_scada}*nc'))
Scada=xr.open_mfdataset(sorted(files_scada)).compute()
Scada=Scada.rename({'WTUR.DateTime':'time'})
Scada['time']=Scada.time+pd.Timedelta(minutes=5)#SCADA timestamps mark bin start, loads use bin centers


power=Scada[pow_var].where(Scada[pow_flag_var]>=min_data_avail)#kept available for future use
ws=Scada[ws_var].where(power>min_power)

#align wind speed onto the load time grid
ws_aligned=ws.reindex(time=Loads.time.values).values

ws_bins=np.arange(0,np.nanmax(ws_aligned)+ws_bin_width,ws_bin_width)
ws_avg=(ws_bins[1:]+ws_bins[:-1])/2

highlight=(Loads.time.values>=highlight_period[0])&(Loads.time.values<=highlight_period[1])

#%% Main
plt.figure(figsize=(14,5))
for i,v in enumerate(loads_var):
    var=f'{turbine_id}_{v}'
    load=Loads[var]/np.nanmax(Loads[var][highlight])

    ax=plt.subplot(1,len(loads_var),i+1)

    #2D histogram of ws vs. normalized load occurrence, contoured in log10 probability to highlight rare regions
    load_bins=np.arange(0,np.nanmax(load.values)+load_bin_width,load_bin_width)
    valid_hist=np.isfinite(ws_aligned)&np.isfinite(load.values)
    H,xedges,yedges=np.histogram2d(ws_aligned[valid_hist],load.values[valid_hist],bins=[ws_bins,load_bins])
    logP=np.ma.masked_invalid(np.log10(H.T/H.sum()))#log10 probability of occurrence, zero-count bins masked
    xc=(xedges[:-1]+xedges[1:])/2
    yc=(yedges[:-1]+yedges[1:])/2
    cf=ax.contourf(xc,yc,logP,levels=20,cmap='Greys')
    plt.colorbar(cf,ax=ax,label='log$_{10}$(Probability of occurrence)',pad=0.02)
    
    #highlight period plotted on top as an oriented (arrow-connected) trajectory
    ws_highlight=ws_aligned[highlight]
    load_highlight=load.values[highlight]
    times_highlight=pd.to_datetime(Loads.time.values[highlight])
    for j in range(len(ws_highlight)-1):
        if ~np.isnan(ws_highlight[j]+load_highlight[j]+ws_highlight[j+1]+load_highlight[j+1]):
            arrow=FancyArrowPatch((ws_highlight[j],load_highlight[j]),(ws_highlight[j+1],load_highlight[j+1]),
                                   arrowstyle='->',color='r',mutation_scale=15,zorder=10)
            ax.add_patch(arrow)
    plt.scatter(ws_highlight,load_highlight,s=30,color='r',zorder=11,label='Selected period')
    # for x,y,t in zip(ws_highlight,load_highlight,times_highlight):
    #     ax.annotate(t.strftime('%H%M'),(x,y),xytext=(0,5),textcoords='offset points',
    #                 color='r',ha='center',va='bottom',fontsize=9,zorder=11)
    plt.xlabel('$U_h$ [m s$^{-1}$]')
    plt.ylabel(labels[v])
    plt.grid()
    if i==0:
        plt.legend()

plt.tight_layout()

os.makedirs(os.path.join(cd,'figures/DEL_stats'),exist_ok=True)
plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.DEL_stats.png'))

