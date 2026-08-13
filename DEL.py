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
matplotlib.rcParams['font.size'] = 13

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source=os.path.join(cd,'data/awaken/kp.turbine.z03.b0/*20230805*nc')
# source_all=os.path.join(cd,'data/awaken/kp.turbine.z03.del/*nc')
loads_var=['tb_bend_foreaft','tb_bend_sideside','tb_bend_resultant','b1_bend_flap_root','b1_bend_edge_root','b1_bend_root_resultant']
bins_time=np.arange(np.datetime64('2023-08-05T10:00:00'),
                    np.datetime64('2023-08-05T12:01:00'),
                    np.timedelta64(600,'s'))

m={'tb_bend_resultant':3,'b1_bend_root_resultant':10}#Mahler exponent
turbine_id='e6'

#graphics
cmap = plt.cm.RdYlGn_r
labels={'tb_bend_foreaft':r'$M_{tb,fa}$ [kNm]',
        'tb_bend_sideside':r'$M_{tb,ss}$ [kNm]',
        'tb_bend_resultant':r'$M_{tb,res}$ [kNm]',
        'b1_bend_root_resultant':r'$M_{br,res}$ [kNm]',
        'b1_bend_flap_root':r'$M_{br,fl}$ [kNm]',
        'b1_bend_edge_root':r'$M_{br,ed}$ [kNm]'}


norm={'tb_bend_foreaft':'tb_bend_resultant',
      'tb_bend_sideside':'tb_bend_resultant',
      'b1_bend_flap_root':'b1_bend_root_resultant',
      'b1_bend_edge_root':'b1_bend_root_resultant'}

#%% Initialization
files=np.array(sorted(glob.glob(source)))
turbine_ids=np.array([f.split('.')[-2] for f in files])

#read long-term DELs
# files_all=glob.glob(source_all)
# DEL_all=xr.open_mfdataset(files_all).compute()
loads_var_tur=[f'{turbine_id}_{v}' for v in loads_var]
time_avg=bins_time[:-1]+(bins_time[1:]-bins_time[:-1])/2
DEL=xr.Dataset()

Data=xr.open_mfdataset(files[turbine_ids==turbine_id])
Data_qc=xr.Dataset()

#read data
for v in loads_var_tur:
    if v in Data.data_vars:
        Data_qc[v]=Data[v].where(Data['qc_'+v]==0).compute()

#%% Main
plt.figure(figsize=(18,8))
ctr=1
for v in loads_var_tur:
    ax=plt.subplot(len(norm),1,ctr)
    if v in Data_qc.data_vars and 'resultant' not in v:
        _del=[]
        max_L=np.float64(Data_qc[turbine_id+'_'+norm[v[3:]]].max())
        for t1,t2 in zip(bins_time[:-1],bins_time[1:]):
            Data_sel=Data_qc.where((Data_qc.time>t1)*(Data_qc.time<t2),drop=True)#selected time bin
            # time=(Data_sel.time.values-Data_sel.time.values[0])/np.timedelta64(1,'s')#time in s
            
            L=Data_sel[v].values#qc
            
            # _del =np.append(_del, equivalent_load(time, L, m=m[v]))#DEL calculation
            # del_quantile=np.float64(np.round(np.sum(_del[-1]>DEL_all[f'{turbine_id}_{v}'])/np.sum(~np.isnan(DEL_all[f'{turbine_id}_{v}']))*100,2))
            
            # color=cmap((del_quantile-30)/70)
            # darker_color = tuple([x * 0.6 if i < 3 else x for i, x in enumerate(color)])
            # plt.fill_between([t1,t2],[0,0],[1.1,1.1],color=color,alpha=0.25)
            # plt.text(t1+(t2-t1)*0.5,1.05,s=str(del_quantile)+'%',color=darker_color,
            #          fontweight='bold',ha='center',va='center')
            plt.plot(Data_sel.time,L/max_L,'k')
            
        # DEL[f'{turbine_id}_{v}']=xr.DataArray(_del,coords={'time':time_avg})
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xlim([bins_time[0],bins_time[-1]+np.timedelta64(1,'s')])
        if ctr==1:
            ax.set_xticklabels([])
        plt.ylabel(labels[v[3:]])
        
        plt.grid()
        ctr+=1
            
plt.xlabel('Time (UTC)')

# #%% Plots          
# plt.figure(figsize=(14,4))
# ctr=1
# for v in loads_var:
#     ax=plt.subplot(1,len(loads_var),ctr)
#     plt.hist(DEL_all[f'{turbine_id}_{v}'],bins=50,color=(0,0,0,0.5))
#     plt.plot([DEL[f'{turbine_id}_{v}'].max(),DEL[f'{turbine_id}_{v}'].max()],[0,ax.get_ylim()[1]],'--r')
#     print(f'{np.float64(np.round(np.sum(DEL_all[f"{turbine_id}_{v}"]>DEL[f"{turbine_id}_{v}"].max())/np.sum(~np.isnan(DEL_all[f"{turbine_id}_{v}"]))*100,2))}% greater DELs of {v}')
#     plt.xlabel(labels[v])
#     if ctr==1:
#         plt.ylabel('Counts')
#     plt.grid()
#     ctr+=1
    
# fig=plt.figure(figsize=(18,10))
# ctr=1
# for v in loads_var:
#     ax=plt.subplot(len(loads_var),1,ctr)
#     plt.plot(DEL_all.time,DEL_all[f'{turbine_id}_{v}'],'k')
#     plt.plot(DEL.time,DEL[f'{turbine_id}_{v}'],'r')
#     plt.grid()
#     plt.ylabel(labels[v])
#     ctr+=1
# plt.xlabel('Time (UTC)')