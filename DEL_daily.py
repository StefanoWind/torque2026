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
source=os.path.join(cd,'data/awaken/kp.turbine.z03.b0')
loads_var=['tb_bend_resultant','b1_bend_root_resultant']
turbine_id='e6'

sdate='2023-08-01T00:00:00'#start date
edate='2023-09-01T00:00:00'#end date
dt=600#[s] time step

m={'tb_bend_resultant':3,'b1_bend_root_resultant':10}#Mahler exponent

#graphics
cmap = plt.cm.RdYlGn_r
labels={'tb_bend_resultant':'DEL of tower-base bending moment [kNm]',
        'b1_bend_root_resultant':'DEL of blade-root bending moment [kNm]'}

#%% Initialization
dates=np.arange(np.datetime64(sdate),np.datetime64(edate)+np.timedelta64(1,'s'),np.timedelta64(1,'D'))
os.makedirs(os.path.join(source.replace('b0','del')),exist_ok=True)

loads_var_tur=[f'{turbine_id}_{v}' for v in loads_var]
qc_var_tur=[f'qc_{v}' for v in loads_var_tur]

#%% Main
for d in dates:
    
    #time bins
    bins_time=np.arange(d,
                    d+np.timedelta64(1,'D')+np.timedelta64(1,'s'),
                    np.timedelta64(dt,'s'))
    time_avg=bins_time[:-1]+(bins_time[1:]-bins_time[:-1])/2
    
    #find files
    d_str=str(d).split('T')[0].replace('-','')
    files=np.array(sorted(glob.glob(os.path.join(source,'*'+d_str+'*'))))
    
    if len(files)>0:
        turbine_ids=np.array([f.split('.')[-2] for f in files])
        DEL=xr.Dataset()
        
        #read all loads
        Data=xr.open_mfdataset(files[turbine_ids==turbine_id])
        Data=Data[loads_var_tur+qc_var_tur].compute()
        
        plt.figure(figsize=(20,10))
        ctr=1
        for v in loads_var_tur:
            ax=plt.subplot(len(loads_var),1,ctr)
            if v in Data.data_vars:
                _del=[]
                for t1,t2 in zip(bins_time[:-1],bins_time[1:]):
                    Data_sel=Data.where((Data.time>t1)*(Data.time<t2),drop=True)#select time bin
                    
                    if len(Data_sel.time)>0:
                        time=(Data_sel.time.values-Data_sel.time.values[0])/np.timedelta64(1,'s')#time in seconds
                        
                        L=Data_sel[v].where(Data_sel[f'qc_{v}']==0).values#qc
                        
                        _del =np.append(_del, equivalent_load(time, L, m=m[v[3:]]))#calculcate del
                        print(f'Calculated DEL at {str(t1).replace("T"," ")}')
                    else:
                        _del =np.append(_del, np.nan)
                
                #store data
                DEL[v]=xr.DataArray(_del,coords={'time':time_avg})
                
                plt.plot(DEL.time,DEL[v],'.-k')
                ax.set_xticklabels([])
                plt.ylabel(labels[v[3:]])
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.grid()
            ctr+=1
        
        #output            
        plt.xlabel('Time (UTC)')
        plt.savefig(os.path.join(source.replace('b0','del'),os.path.basename(source).replace('b0','del')+'.'+d_str+'.png'))
        plt.close()
        
        DEL.to_netcdf(os.path.join(source.replace('b0','del'),os.path.basename(source).replace('b0','del')+'.'+d_str+'.nc'))
        
    else:
        print(f'No files on {d_str}')
                
