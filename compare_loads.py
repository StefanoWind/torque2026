# -*- coding: utf-8 -*-
"""
Compare DELs and max loads between kp.turbine.z03.c0 and kp.turbine.z03.c1
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
source_c0=os.path.join(cd,'data/awaken/kp.turbine.z03.c0/*{turbine_id}.nc')
source_c1=os.path.join(cd,'data/awaken/kp.turbine.z03.c1/*nc')
turbine_id='e6'
channels=['tb_bend_foreaft','tb_bend_sideside', 'b1_bend_flap_root', 'b1_bend_edge_root']
stats=['del','max']

labels={'tb_bend_foreaft_del':r'DEL($M_{tb,fa}$) [kNm]',
        'tb_bend_sideside_del':r'DEL($M_{tb,ss}$) [kNm]',
        'b1_bend_flap_root_del':r'DEL($M_{br,fl}$) [kNm]',
        'b1_bend_edge_root_del':r'DEL($M_{br,ed}$) [kNm]',
        'tb_bend_foreaft_max':r'MAX($M_{tb,fa}$) [kNm]',
        'tb_bend_sideside_max':r'MAX($M_{tb,ss}$) [kNm]',
        'b1_bend_flap_root_max':r'MAX($M_{br,fl}$) [kNm]',
        'b1_bend_edge_root_max':r'MAX($M_{br,ed}$) [kNm]'}

#%% Load data
vars_c0=[f'{turbine_id}_{c}' for c in channels]
ds=[]
for f in tqdm(sorted(glob.glob(source_c0.format(turbine_id=turbine_id))),desc='Loading c0 files'):
    try:
        D=xr.open_dataset(f)
        ds.append(xr.Dataset({f'{v}_{s}':D[v].sel(stat=s,drop=True).where(D[f'qc_{v}'].sel(stat=s,drop=True)==0)
                              for v in vars_c0 for s in stats}).load())
    except:
        print(f'Error loading {os.path.basename(f)}')
C0=xr.concat(ds,dim='time')
C0['time']=C0.time+pd.Timedelta(minutes=5)#c0 timestamps mark bin start, c1 uses bin centers


vars_c1=[f'{turbine_id}_{c}_{s}' for c in channels for s in stats]
ds=[]
for f in tqdm(sorted(glob.glob(source_c1)),desc='Loading c1 files'):
    try:
        ds.append(xr.open_dataset(f)[vars_c1].load())
    except:
        print(f'Error loading {os.path.basename(f)}')
C1=xr.concat(ds,dim='time')

C0,C1=xr.align(C0,C1,join='outer')

#%% Statistics
for v in vars_c1:
    x=C0[v].values
    y=C1[v].values
    real=~np.isnan(x+y)
    print(f'{v}: c0 valid={np.sum(~np.isnan(x))}, c1 valid={np.sum(~np.isnan(y))}, common={np.sum(real)}')
    if np.sum(real)>1:
        bias=np.mean(y[real]-x[real])
        rmse=np.mean((y[real]-x[real])**2)**0.5
        rho=np.corrcoef(x[real],y[real])[0,1]
        print(f'   bias (c1-c0)={bias:.2f} kNm, RMSE={rmse:.2f} kNm, r={rho:.4f}')

#%% Plots
os.makedirs(os.path.join(cd,'figures/compare_loads'),exist_ok=True)

for s in stats:
    vars_s=[f'{turbine_id}_{c}_{s}' for c in channels]
    fig=plt.figure(figsize=(18,12))
    gs=fig.add_gridspec(len(vars_s),2,width_ratios=[4,1])
    for i,v in enumerate(vars_s):
        v_lab=v.replace(f'{turbine_id}_','')
        ax=fig.add_subplot(gs[i,0])
        ax.plot(C0.time,C0[v],'.k',markersize=2,label='c0')
        ax.plot(C1.time,C1[v],'.r',markersize=2,alpha=0.5,label='c1')
        ax.set_ylabel(labels[v_lab])
        ax.grid()
        ax.set_yscale('symlog')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if i==0:
            ax.legend()
        if i<len(vars_s)-1:
            ax.set_xticklabels([])

        ax=fig.add_subplot(gs[i,1])
        ax.plot(C0[v],C1[v],'.k',markersize=2,alpha=0.25)
        lim=[np.nanmin([np.nanpercentile(C0[v],1),np.nanpercentile(C1[v],1)]),np.nanmax([np.nanpercentile(C0[v],99),np.nanpercentile(C1[v],99)])]
        if np.all(np.isfinite(lim)):
            ax.plot(lim,lim,'--r')
        ax.set_xlabel('c0')
        ax.set_ylabel('c1')
        ax.set_aspect('equal')
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(cd,'figures/compare_loads',f'{turbine_id}.compare_loads.{s}.png'))
