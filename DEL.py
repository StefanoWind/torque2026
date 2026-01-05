# -*- coding: utf-8 -*-
"""
Test DEL calculation
"""
import os
cd=os.getcwd()
import numpy as np
from openfast_toolbox.tools.fatigue import equivalent_load
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import glob
import xarray as xr
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/awaken/kp.turbine.z03.b0/*nc')
source_all=os.path.join(cd,'data/awaken/kp.turbine.z03.del/*nc')
loads_var=['tb_bend_resultant','b1_bend_root_resultant']
bins_time=np.arange(np.datetime64('2023-08-05T10:00:00'),
                    np.datetime64('2023-08-05T12:01:00'),
                    np.timedelta64(600,'s'))

m={'tb_bend_resultant':3,'b1_bend_root_resultant':10}#Mahler exponent
turbine_id='e6'

#graphics
cmap = plt.cm.RdYlGn_r
labels={'tb_bend_resultant':'DEL of tower-base bending moment [kNm]',
        'b1_bend_root_resultant':'DEL of blade-root bending moment [kNm]'}

#%% Initialization
files=np.array(sorted(glob.glob(source)))
turbine_ids=np.array([f.split('.')[-2] for f in files])

#read long-term DELs
files_all=glob.glob(source_all)
DEL_all=xr.open_mfdataset(files_all).compute()

time_avg=bins_time[:-1]+(bins_time[1:]-bins_time[:-1])/2
DEL=xr.Dataset()

#%% Main

#read data
Data=xr.open_mfdataset(files[turbine_ids==turbine_id]).compute()


plt.figure(figsize=(20,10))
ctr=1
for v in loads_var:
    ax=plt.subplot(len(loads_var),1,ctr)
    if f'{turbine_id}_{v}' in Data.data_vars:
        _del=[]
        max_v=np.round(np.max(Data[f'{turbine_id}_{v}'].where(Data[f'qc_{turbine_id}_{v}']==0).values)/5000)*5000+2500
        for t1,t2 in zip(bins_time[:-1],bins_time[1:]):
            Data_sel=Data.where((Data.time>t1)*(Data.time<t2),drop=True)#selected time bin
            time=(Data_sel.time.values-Data_sel.time.values[0])/np.timedelta64(1,'s')#time in s
            
            L=Data_sel[f'{turbine_id}_{v}'].where(Data_sel[f'qc_{turbine_id}_{v}']==0).values#qc
            
            _del =np.append(_del, equivalent_load(time, L, m=m[v]))#DEL calculation
           
            color=cmap(_del[-1]/4000)
            darker_color = tuple([x * 0.6 if i < 3 else x for i, x in enumerate(color)])
            plt.fill_between([t1,t2],[0,0],[max_v,max_v],color=color,alpha=0.25)
            plt.text(t1+(t2-t1)*0.5,max_v*0.95,s=int(_del[-1]),color=darker_color,
                     fontweight='bold',ha='center',va='center')
            plt.plot(Data_sel.time,L,'k')
            
        DEL[f'{turbine_id}_{v}']=xr.DataArray(_del,coords={'time':time_avg})
        ax.set_xticklabels([])
        
        plt.ylabel(labels[v])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.grid()
    ctr+=1
            
plt.xlabel('Time (UTC)')

#%% Plots
            
plt.figure(figsize=(14,6))
ctr=1
for v in loads_var:
    ax=plt.subplot(1,len(loads_var),ctr)
    plt.hist(DEL_all[f'{turbine_id}_{v}'],bins=100,color=(0,0,0,0.5),density=True)
    plt.plot([DEL[f'{turbine_id}_{v}'].max(),DEL[f'{turbine_id}_{v}'].max()],[0,1],'--r')
    print(f'{np.float64(np.round(np.sum(DEL_all[f"{turbine_id}_{v}"]>DEL_all[f"{turbine_id}_{v}"])/np.sum(~np.isnan(DEL_all[f"{turbine_id}_{v}"]))*100,2))} greater DELs of {v}')
    plt.xlabel(labels[v])
    if ctr==1:
        plt.ylabel('Probability')
    else:
        ax.set_yticklabels([])
    plt.grid()
    ctr+=1