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
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_loads=os.path.join(cd,'data/awaken/kp.turbine.z03.c1/*nc')
source_scada=os.path.join(cd,'data/awaken/kp.turbine.z02.c0')
turbine_id='e6'
turbine_id_scada='wt041'#SCADA turbine id corresponding to turbine_id (Alias E06)
loads_var=['tb_bend_resultant_del','b1_bend_root_resultant_del',
           'tb_bend_resultant_max','b1_bend_root_resultant_max']
ws_var='WMET.HorWdSpd_10m_Avg'
ti_var='WMET.HorWdSpd_10m_Std'
min_data_avail=0.25
min_ti=1
dws=1
dti=2
ws_bins=np.arange(0,25,dws)
ti_bins=np.append(np.arange(5,30,dti),100)
load_bins=np.arange(0,1.31,0.1)

max_ci_frac=0.1#max width of the bootstrap CI, as a fraction of the load's std

highlight_period=[np.datetime64('2023-08-05T10:00:00'),np.datetime64('2023-08-05T12:00:00')]#datetimes highlighted in the scatter

labels={'tb_bend_foreaft_del':r'DEL($M_{tb,fa}$) [kNm]',
        'tb_bend_sideside_del':r'DEL($M_{tb,ss}$) [kNm]',
        'tb_bend_resultant_del':r'DEL($M_{tb,res}$) [kNm]',
        'b1_bend_root_resultant_del':r'DEL($M_{br,res}$) [kNm]',
        'b1_bend_flap_root_del':r'DEL($M_{br,fl}$) [kNm]',
        'b1_bend_edge_root_del':r'DEL($M_{br,ed}$) [kNm]',
        'tb_bend_foreaft_max':r'MAX($M_{tb,fa}$) [kNm]',
        'tb_bend_sideside_max':r'MAX($M_{tb,ss}$) [kNm]',
        'tb_bend_resultant_max':r'MAX($M_{tb,res}$) [kNm]',
        'b1_bend_root_resultant_max':r'MAX($M_{br,res}$) [kNm]',
        'b1_bend_flap_root_max':r'MAX($M_{br,fl}$) [kNm]',
        'b1_bend_edge_root_max':r'MAX($M_{br,ed}$) [kNm]'}

#%% Load data

files=sorted(glob.glob(source_loads))

ds=[]
for f in tqdm(files,desc='Loading loads files'):
    try:
        ds.append(xr.open_dataset(f).load())
    except:
        print(f'Error loading {os.path.basename(f)}')
Loads=xr.concat(ds,dim='time')
                 
#restrict SCADA files to the days actually present in the loads dataset
dates=pd.DatetimeIndex(np.unique(Loads.time.dt.floor('D').values))
files_scada=[]
for d in dates:
    files_scada+=glob.glob(os.path.join(source_scada,f'*{d.strftime("%Y%m%d")}*{turbine_id_scada}*nc'))
scada_vars=[ws_var,ti_var]
Scada=xr.concat([xr.open_dataset(f)[scada_vars].load() for f in tqdm(sorted(files_scada),desc='Loading SCADA files')],dim='WTUR.DateTime')
Scada=Scada.rename({'WTUR.DateTime':'time'})
Scada['time']=Scada.time+pd.Timedelta(minutes=5)#SCADA timestamps mark bin start, loads use bin centers

ws=Scada[ws_var]
ti=Scada[ti_var]/Scada[ws_var]*100
ti=ti.where(ti>min_ti)

#align wind speed onto the load time grid
ws_aligned=ws.reindex(time=Loads.time.values).values
ti_aligned=ti.reindex(time=Loads.time.values).values

highlight=(Loads.time.values>=highlight_period[0])&(Loads.time.values<=highlight_period[1])

#%% PDF
#joint probability mass function f(TI, WS, load), computed once per load channel from a single 3D histogram
os.makedirs(os.path.join(cd,'figures/DEL_stats'),exist_ok=True)

ws_avg=  (ws_bins[:-1]+ws_bins[1:]    )/2
ti_avg=  (ti_bins[1:]+ti_bins[:-1]    )/2
load_avg=(load_bins[1:]+load_bins[:-1])/2

load_dict={}
pdf_3d={}
for v in loads_var:
    var=f'{turbine_id}_{v}'
    load=Loads[var]/np.nanmax(Loads[var][highlight])
    load_dict[v]=load

    valid=np.isfinite(ti_aligned)&np.isfinite(ws_aligned)&np.isfinite(load.values)
    H,_=np.histogramdd(np.column_stack([ti_aligned[valid],ws_aligned[valid],load.values[valid]]),
                        bins=[ti_bins,ws_bins,load_bins])
    pdf_3d[v]=H/H.sum()#normalized so that sum(pdf)=1

#joint pdf f(TI, WS), independent of load channel, plotted as a shadow on the WS-TI (load=0) plane
valid_2d=np.isfinite(ti_aligned)&np.isfinite(ws_aligned)
H2,_,_=np.histogram2d(ti_aligned[valid_2d],ws_aligned[valid_2d],bins=[ti_bins,ws_bins])
pdf_2d=H2/H2.sum()#normalized so that sum(pdf)=1
logP2=np.ma.masked_invalid(np.log10(pdf_2d).T)#zero-density bins masked


prob={}
prob_cond={}
n_highlight=len(np.where(highlight)[0])
for v in loads_var:
    var=f'{turbine_id}_{v}'
    prob[v]=     np.zeros(n_highlight)
    prob_cond[v]=np.zeros(n_highlight)

    for i,it in enumerate(np.where(highlight)[0]):
        ws_sel=ws_aligned[it]
        ti_sel=ti_aligned[it]
        L=Loads[var]/np.nanmax(Loads[var][highlight])
        L_sel=np.float64(L.isel(time=it))     
        if np.isfinite(ws_sel) and np.isfinite(ti_sel) and np.isfinite(L_sel): 
               
            i_ws=np.where(ws_bins>ws_sel)[0][0]-1
            i_ti=np.where(ti_bins>ti_sel)[0][0]-1
            sel_ws=(ws_aligned>ws_bins[i_ws])*(ws_aligned<=ws_bins[i_ws+1])
            sel_ti=(ti_aligned>ti_bins[i_ti])*(ti_aligned<=ti_bins[i_ti+1])
            
            prob[v][i]=np.sum(L>=L_sel)/np.sum(np.isfinite(L))
            prob_cond[v][i]=np.sum(L[sel_ws*sel_ti]>=L_sel)/np.sum(np.isfinite(L[sel_ws*sel_ti]))
        else:
            prob[v][i]=np.nan
            prob_cond[v][i]=np.nan

#%% Prob/prob_cond history
plt.close('all')

time_highlight=Loads.time.values[highlight]

fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True)
ax=ax.flatten()
for i,v in enumerate(loads_var):
    ax[i].plot(time_highlight,prob[v],'.-k',label='$P$')
    ax[i].plot(time_highlight,prob_cond[v],'.-r',label='$P_{cond}$')
    ax[i].set_ylim([0,1])
    ax[i].set_ylabel(labels[v])
    ax[i].grid(True)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[0].legend()
for a in ax[2:]:
    a.set_xlabel('Time (UTC)')
plt.tight_layout()
plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.prob_history.png'))

#%% Main (filled-slice style, colored by pdf)
plt.close('all')
perc_levels=[0.05,0.1,0.2,0.5,1,2,5]#[%] probability levels shown as filled contour bands, increasing order
levels=np.log10(np.array(perc_levels)/100)

for i,v in enumerate(loads_var):
    load=load_dict[v]
    yc=(load_bins[:-1]+load_bins[1:])/2
    pdf=pdf_3d[v]

    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection='3d')

    #pdf slices stacked along the TI axis
    for k in range(len(ti_bins)-1):
        logP=np.ma.masked_invalid(np.log10(pdf[k]).T)#zero-density bins masked
        cf=ax.contourf(logP,ws_avg,load_avg,zdir='x',offset=ti_avg[k],levels=levels,cmap='Greys',extend='max',alpha=0.5)
        ax.contour(    logP,ws_avg,load_avg,zdir='x',offset=ti_avg[k],levels=levels,
                   extend='max',colors='k',linestyles='solid',zorder=11,linewidths=0.5)
        
    #highlight period plotted as a 3D (TI, WS, load) trajectory
    valid_traj=highlight&np.isfinite(ws_aligned)&np.isfinite(load.values)&np.isfinite(ti_aligned)
    ws_highlight=ws_aligned[valid_traj]
    load_highlight=load.values[valid_traj]
    ti_highlight=ti_aligned[valid_traj]
    ax.plot(ti_highlight,ws_highlight,load_highlight,color='r',linewidth=1,zorder=10)
    # ax.quiver(ti_highlight[:-1],ws_highlight[:-1],load_highlight[:-1],
    #           np.diff(ti_highlight),np.diff(ws_highlight),np.diff(load_highlight),
    #           color='k',arrow_length_ratio=0.01,normalize=False,linewidth=0.75,zorder=11,linewidths=0.5)
    ax.scatter(ti_highlight,ws_highlight,load_highlight,s=20,color='r',zorder=11,label='Selected period',alpha=1)

    # ax.set_xscale('symlog')
    ax.set_xlim([ti_bins[0],ti_bins[-1]])
    ax.set_ylim([ws_bins[0],ws_bins[-1]])
    ax.set_zlim([0,1.2])
    ax.set_box_aspect((2,1,1))#stretch the TI axis for readability
    ax.view_init(elev=40,azim=-140,roll=0)
    ax.set_xlabel('TI [%]')
    ax.set_ylabel('$U_h$ [m s$^{-1}$]')
    ax.set_zlabel(labels[v])
    ax.legend()

    cbar=fig.colorbar(cf,ax=ax,label='Probability of occurrence',pad=0.1,shrink=0.6,ticks=levels)
    cbar.ax.set_yticklabels([f'{p:g}%' for p in perc_levels])
    # plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.{v}.DEL_stats_3D.png'))

