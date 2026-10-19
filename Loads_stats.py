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
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_loads=os.path.join(cd,'data/awaken/kp.turbine.z03.c0/*{turbine_id}.nc')
source_scada=os.path.join(cd,'data/awaken/kp.turbine.z02.c0')
turbine_id='e6'
turbine_id_scada='wt041'#SCADA turbine id corresponding to turbine_id (Alias E06)
loads_var=['tb_bend_foreaft_del','b1_bend_flap_root_del',
           'tb_bend_foreaft_max','b1_bend_flap_root_max']#c0 resultant DELs are all QC-flagged
ws_var='WMET.HorWdSpd_10m_Avg'
ti_var='WMET.HorWdSpd_10m_Std'
yaw_spt_min_var='WYAW.YwSpt_10m_Min'
yaw_spt_max_var='WYAW.YwSpt_10m_Max'
wake_steering_start='2024-05-01'
min_data_avail=0.25
min_ti=1
dws=1
dti=2
max_yaw_spt=1
yaw_spt_missing=999#[deg] assigned to missing yaw setpoints after wake_steering_start so those bins are excluded
ws_bins=np.arange(0,25,dws)
ti_bins=np.append(np.arange(6,31,dti),100)
load_bins=np.arange(0,1.31,0.1)

max_ci_frac=0.1#max width of the bootstrap CI, as a fraction of the load's std

highlight_period=[np.datetime64('2023-08-05T10:00:00'),np.datetime64('2023-08-05T12:00:00')]#datetimes highlighted in the scatter

labels={'tb_bend_foreaft_del':r'DEL($M_{tb,fa}$)',
        'tb_bend_sideside_del':r'DEL($M_{tb,ss}$)',
        'tb_bend_resultant_del':r'DEL($M_{tb,res}$)',
        'b1_bend_root_resultant_del':r'DEL($M_{br,res}$)',
        'b1_bend_flap_root_del':r'DEL($M_{br,fl}$)',
        'b1_bend_edge_root_del':r'DEL($M_{br,ed}$)',
        'tb_bend_foreaft_max':r'MAX($|M_{tb,fa}|$)',
        'tb_bend_sideside_max':r'MAX($|M_{tb,ss}|$)',
        'tb_bend_resultant_max':r'MAX($|M_{tb,res}|$)',
        'b1_bend_root_resultant_max':r'MAX($|M_{br,res}|$)',
        'b1_bend_flap_root_max':r'MAX($|M_{br,fl}|$)',
        'b1_bend_edge_root_max':r'MAX($|M_{br,ed}|$)',
        'tb_bend_foreaft_min':r'MIN($M_{tb,fa}$)',
        'tb_bend_sideside_min':r'MIN($M_{tb,ss}$)',
        'tb_bend_resultant_min':r'MIN($M_{tb,res}$)',
        'b1_bend_root_resultant_min':r'MIN($M_{br,res}$)',
        'b1_bend_flap_root_min':r'MIN($M_{br,fl}$)',
        'b1_bend_edge_root_min':r'MIN($M_{br,ed}$)'}

#%% Load data

files=sorted(glob.glob(source_loads.format(turbine_id=turbine_id)))

ds=[]
for f in tqdm(files,desc='Loading loads files'):
    try:
        D=xr.open_dataset(f)
        L={}
        for v in loads_var:
            channel,stat=v.rsplit('_',1)
            var=f'{turbine_id}_{channel}'
            if stat=='max':
                #max load as the largest absolute value between 10-min max and min [Robertson et al., 2019]; NaN if either is flagged
                x={s:D[var].sel(stat=s,drop=True).where(D[f'qc_{var}'].sel(stat=s,drop=True)==0) for s in ['max','min']}
                L[f'{turbine_id}_{v}']=np.fmax(np.abs(x['max']),np.abs(x['min']))
            else:
                L[f'{turbine_id}_{v}']=D[var].sel(stat=stat,drop=True).where(D[f'qc_{var}'].sel(stat=stat,drop=True)==0)
        ds.append(xr.Dataset(L).load())
    except:
        print(f'Error loading {os.path.basename(f)}')
Loads=xr.concat(ds,dim='time')
Loads['time']=Loads.time+pd.Timedelta(minutes=5)#c0 timestamps mark bin start, SCADA is shifted to bin centers below
                 
#restrict SCADA files to the days actually present in the loads dataset
dates=pd.DatetimeIndex(np.unique(Loads.time.dt.floor('D').values))
files_scada=[]
for d in dates:
    files_scada+=glob.glob(os.path.join(source_scada,f'*{d.strftime("%Y%m%d")}*{turbine_id_scada}*nc'))
scada_vars=[ws_var,ti_var,yaw_spt_min_var,yaw_spt_max_var]

def read_scada(f):
    D=xr.open_dataset(f)
    for v in [yaw_spt_min_var,yaw_spt_max_var]:
        if v not in D:
            D[v]=xr.full_like(D[ws_var],np.nan)#older files lack the yaw setpoint channels
    return D[scada_vars].load()

Scada=xr.concat([read_scada(f) for f in tqdm(sorted(files_scada),desc='Loading SCADA files')],dim='WTUR.DateTime')
Scada=Scada.rename({'WTUR.DateTime':'time'})
Scada['time']=Scada.time+pd.Timedelta(minutes=5)#SCADA timestamps mark bin start, loads use bin centers

ws=Scada[ws_var]
ti=Scada[ti_var]/Scada[ws_var]*100
ti=ti.where(ti>min_ti)

#exclude loads from 10-min bins with a nonzero yaw setpoint
#missing setpoints (no file, no variable or NaN) are assumed 0 before wake steering started and excluded after
after_ws_start=xr.DataArray(Loads.time.values>=np.datetime64(wake_steering_start),coords={'time':Loads.time.values})
yaw_spt_fill=xr.where(after_ws_start,yaw_spt_missing,0)
yaw_spt_min=Scada[yaw_spt_min_var].reindex(time=Loads.time.values)
yaw_spt_max=Scada[yaw_spt_max_var].reindex(time=Loads.time.values)
yaw_spt_isnan=yaw_spt_min.isnull()|yaw_spt_max.isnull()
yaw_spt_min=yaw_spt_min.fillna(yaw_spt_fill)
yaw_spt_max=yaw_spt_max.fillna(yaw_spt_fill)
yaw_spt_exceed=(np.abs(yaw_spt_min)>max_yaw_spt)|(np.abs(yaw_spt_max)>max_yaw_spt)
Loads=Loads.where(~yaw_spt_exceed)

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

prob={}
prob_cond={}
n_exceed={}
n_total={}
n_exceed_cond={}
n_total_cond={}
n_highlight=len(np.where(highlight)[0])
for v in loads_var:
    var=f'{turbine_id}_{v}'
    n_exceed[v]=     np.full(n_highlight,np.nan)
    n_total[v]=      np.full(n_highlight,np.nan)
    n_exceed_cond[v]=np.full(n_highlight,np.nan)
    n_total_cond[v]= np.full(n_highlight,np.nan)

    for i,it in enumerate(np.where(highlight)[0]):
        ws_sel=ws_aligned[it]
        ti_sel=ti_aligned[it]
        L=load_dict[v]
        L_sel=np.float64(L.isel(time=it))     
        if np.isfinite(ws_sel) and np.isfinite(ti_sel) and np.isfinite(L_sel): 
               
            i_ws=np.where(ws_bins>ws_sel)[0][0]-1
            i_ti=np.where(ti_bins>ti_sel)[0][0]-1
            sel_ws=(ws_aligned>ws_bins[i_ws])*(ws_aligned<=ws_bins[i_ws+1])
            sel_ti=(ti_aligned>ti_bins[i_ti])*(ti_aligned<=ti_bins[i_ti+1])
            
            exceed=L>=L_sel
            n_exceed[v][i]=     np.sum(exceed)
            n_total[v][i]=      np.sum(np.isfinite(L))
            n_exceed_cond[v][i]=np.sum(exceed[sel_ws*sel_ti])
            n_total_cond[v][i]= np.sum(np.isfinite(L[sel_ws*sel_ti]))

    prob[v]=     n_exceed[v]/n_total[v]
    prob_cond[v]=n_exceed_cond[v]/n_total_cond[v]

#%% Plots

# Prob/prob_cond history

plt.close('all')

time_highlight=Loads.time.values[highlight]

fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True)
ax=ax.flatten()
for i,v in enumerate(loads_var):
    ax[i].plot(time_highlight,prob[v],'o-',color='#1b3a6b',markersize=14,label='$P$')
    ax[i].plot(time_highlight,prob_cond[v],'o-',color='#d4a017',markersize=14,label='$P_{cond}$')
    #sample count of the WS-TI bin shown inside the marker only where it is small
    for t,p,nt in zip(time_highlight,prob_cond[v],n_total_cond[v]):
        if np.isfinite(p) and nt<100:
            ax[i].text(t,p,f'{nt:.0f}',ha='center',va='center',fontsize=9,color='k',fontweight='bold',zorder=10)
    ax[i].set_ylim([0,1.1])
    ax[1].set_yticks([0,0.25,0.5,0.75,1])
    ax[i].set_ylabel(labels[v])
    ax[i].grid(True)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[0].legend()
for a in ax[2:]:
    a.set_xlabel('Time (UTC)')
plt.tight_layout()
# plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.prob_history.png'))

#% 3D PDF
perc_levels=[0.05,0.1,0.2,0.5,1,2,5]#[%] probability levels shown as filled contour bands, increasing order
levels=np.log10(np.array(perc_levels)/100)
ti_high_plot=31#[%] TI plane where the open-ended last TI bin (pdf slice and trajectory points) is drawn
ti_offset=np.append(ti_avg[:-1],ti_high_plot)

for stat in ['del','max']:
    vars_stat=[v for v in loads_var if v.endswith(f'_{stat}')]
    fig=plt.figure(figsize=(18,10))
    axs=[fig.add_subplot(1,len(vars_stat),j+1,projection='3d') for j in range(len(vars_stat))]

    for ax,v in zip(axs,vars_stat):
        load=load_dict[v]
        pdf=pdf_3d[v]

        #pdf slices stacked along the TI axis
        for k in range(len(ti_bins)-1):
            logP=np.ma.masked_invalid(np.log10(pdf[k]).T)#zero-density bins masked
            cf=ax.contourf(logP,ws_avg,load_avg,zdir='x',offset=ti_offset[k],levels=levels,cmap='Greys',extend='max',alpha=0.5)
            ax.contour(    logP,ws_avg,load_avg,zdir='x',offset=ti_offset[k],levels=levels,
                       extend='max',colors='k',linestyles='solid',zorder=11,linewidths=0.5)

        #highlight period plotted as a 3D (TI, WS, load) trajectory
        valid_traj=highlight&np.isfinite(ws_aligned)&np.isfinite(load.values)&np.isfinite(ti_aligned)
        ws_highlight=ws_aligned[valid_traj]
        load_highlight=load.values[valid_traj]
        ti_highlight=ti_aligned[valid_traj]
        ti_highlight=np.where(ti_highlight>ti_bins[-2],ti_high_plot,ti_highlight)
        dtime_highlight=(Loads.time.values[valid_traj]-highlight_period[0])/np.timedelta64(1,'m')
        ax.plot(ti_highlight,ws_highlight,load_highlight,color='k',linewidth=1,zorder=10)
        sc=ax.scatter(ti_highlight,ws_highlight,load_highlight,s=20,c=dtime_highlight,cmap='jet',vmin=0,vmax=120,
                      zorder=11,alpha=1)

        ax.set_xlim([ti_bins[0],ti_high_plot+1])
        ax.set_xticks(np.append(np.arange(10,ti_bins[-2],5),ti_high_plot))
        ax.set_xticklabels([f'{t:g}' for t in np.arange(10,ti_bins[-2],5)]+[f'>{ti_bins[-2]:g}'])
        ax.set_ylim([ws_bins[0],ws_bins[-1]])
        ax.set_zlim([load_bins[0],1.2])
        ax.set_box_aspect((2,1,1))#stretch the TI axis for readability
        ax.view_init(elev=40,azim=-140,roll=0)
        ax.set_xlabel('TI [%]')
        ax.set_ylabel('$U_h$ [m s$^{-1}$]')
        ax.set_zlabel(labels[v])

    #time colorbar created first so it spans the full figure width, before the probability colorbar takes space on the right
    cbar=fig.colorbar(cf,ax=axs,label='Probability of occurrence',location='right',pad=0.05,shrink=0.5,ticks=levels)
    cbar.ax.set_yticklabels([f'{p:g}%' for p in perc_levels])
    fig.colorbar(sc,ax=axs,label=f'Time since {str(highlight_period[0])[11:16]} UTC [min]',
                 location='bottom',pad=0.03,shrink=1,aspect=60,ticks=np.arange(0,121,30))
    # plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.{stat}.DEL_stats_3D.png'))


# Yaw setpoint check
time_all=Loads.time.values
excluded_missing=(yaw_spt_exceed&yaw_spt_isnan&after_ws_start).values
excluded_spt=(yaw_spt_exceed&~yaw_spt_isnan).values

fig,ax=plt.subplots(1,1,figsize=(14,5))
ax.fill_between(time_all,0,1,where=excluded_spt,step='mid',color='r',alpha=0.3,linewidth=0,
                transform=ax.get_xaxis_transform(),label=rf'Excluded ($|$yaw$_{{spt}}|>{max_yaw_spt}^\circ$)')
ax.fill_between(time_all,0,1,where=excluded_missing,step='mid',color='b',alpha=0.3,linewidth=0,
                transform=ax.get_xaxis_transform(),label='Excluded (missing after wake steering start)')
ax.plot(time_all,Scada[yaw_spt_max_var].reindex(time=time_all),'.k',markersize=1,label='Max yaw$_{spt}$')
ax.plot(time_all,Scada[yaw_spt_min_var].reindex(time=time_all),'.g',markersize=1,label='Min yaw$_{spt}$')
ax.axvline(np.datetime64(wake_steering_start),color='k',linestyle='--',label='Wake steering start')
ax.set_xlim([time_all[0],time_all[-1]])
ax.set_ylabel(r'Yaw setpoint [$^\circ$]')
ax.set_xlabel('Time (UTC)')
ax.grid(True)
ax.legend(markerscale=10)
plt.tight_layout()
# plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.yaw_spt_check.png'))
