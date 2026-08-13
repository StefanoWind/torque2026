# -*- coding: utf-8 -*-
"""
Batch DEL calculation: computes damage equivalent loads in 10-min bins, run per day over a date range.

Inputs (both hard-coded and available as command line inputs in this order):
    source: path to the b0 turbine-loads folder
    turbine_id: two-char turbine code
    sdate [%Y-%m-%dT%H:%M:%S]: start date in UTC
    edate [%Y-%m-%dT%H:%M:%S]: end date in UTC
    replace [bool]: whether to overwrite existing daily outputs
    mode [str]: serial or parallel
"""
import os
cd=os.getcwd()
import numpy as np
import sys
from openfast_toolbox.tools.fatigue import equivalent_load
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import glob
import xarray as xr
import warnings
import matplotlib
from multiprocessing import Pool
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
if len(sys.argv)==1:
    source=os.path.join(cd,'data/awaken/kp.turbine.z03.b0')
    turbine_id='e6'
    sdate='2023-08-01T00:00:00'#start date
    edate='2023-09-01T00:00:00'#end date
    replace=False
    mode='serial'#serial or parallel
else:
    source=sys.argv[1]
    turbine_id=sys.argv[2]
    sdate=sys.argv[3]
    edate=sys.argv[4]
    replace=sys.argv[5]=='True'
    mode=sys.argv[6]#serial or parallel

loads_var=['tb_bend_resultant','b1_bend_root_resultant','active_power']

dt=600#[s] time step

m={'tb_bend_resultant':3,'b1_bend_root_resultant':10}#Mahler exponent

#graphics
cmap = plt.cm.RdYlGn_r
labels={'tb_bend_resultant':'DEL of tower-base bending moment [kNm]',
        'b1_bend_root_resultant':'DEL of blade-root bending moment [kNm]',
        'active_power':'Active power [kW]'}

loads_var_tur=[f'{turbine_id}_{v}' for v in loads_var]
qc_var_tur=[f'qc_{v}' for v in loads_var_tur]

#%% Functions
def process_day(d,source,turbine_id,loads_var,loads_var_tur,qc_var_tur,dt,m,labels,replace):

    #time bins
    bins_time=np.arange(d,
                        d+np.timedelta64(1,'D')+np.timedelta64(1,'s'),
                        np.timedelta64(dt,'s'))
    time_avg=bins_time[:-1]+(bins_time[1:]-bins_time[:-1])/2
    
    #find files
    d_str=str(d).split('T')[0].replace('-','')
    files=np.array(sorted(glob.glob(os.path.join(source,'*'+d_str+'*'))))
    
    save_name=os.path.join(source.replace('b0','del'),os.path.basename(source).replace('b0','del')+'.'+d_str+'.nc')
    
    if os.path.isfile(save_name)==False or replace:
        if len(files)>0:
            turbine_ids=np.array([f.split('.')[-2] for f in files])
            DEL=xr.Dataset()
            
            #read all loads
            try:
                Data=xr.open_mfdataset(files[turbine_ids==turbine_id])
                Data=Data[loads_var_tur+qc_var_tur].compute()
                
                plt.figure(figsize=(20,10))
                ctr=1
                for v in loads_var_tur:
                    ax=plt.subplot(len(loads_var),1,ctr)
                    if v in Data.data_vars:
                        _del=[]
                        for t1,t2 in zip(bins_time[:-1],bins_time[1:]):
                            Data_sel=Data.where((Data.time>=t1)*(Data.time<t2),drop=True)#select time bin (half-open, avoids dropping samples on bin edges)
                            
                            if len(Data_sel.time)>0:
                                
                                if m[v[3:]] is None:
                                    _del=np.append(_del, Data_sel[v].where(Data_sel[f'qc_{v}']==0).mean(dim='time').values)
                                else:
                                    time=(Data_sel.time.values-Data_sel.time.values[0])/np.timedelta64(1,'s')#time in seconds
                                    
                                    L=Data_sel[v].where(Data_sel[f'qc_{v}']==0).values#qc
                                    
                                    _del =np.append(_del, equivalent_load(time, L, m=m[v[3:]]))#calculcate del
                                    print(f'Calculated DEL at {str(t1).replace("T"," ")}',flush=True)
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
                
                DEL.to_netcdf(save_name)
            except Exception as e:
                print(f'Error on day {str(d)}')
                print(e)
        else:
            print(f'No files on {d_str}')
    else:
        print(f'{os.path.basename(save_name)} already exists')

#%% Main
if __name__=='__main__':
    dates=np.arange(np.datetime64(sdate),np.datetime64(edate)+np.timedelta64(1,'s'),np.timedelta64(1,'D'))
    os.makedirs(os.path.join(source.replace('b0','del')),exist_ok=True)

    if mode=='serial':
        for d in dates:
            process_day(d,source,turbine_id,loads_var,loads_var_tur,qc_var_tur,dt,m,labels,replace)
    elif mode=='parallel':
        args=[(d,source,turbine_id,loads_var,loads_var_tur,qc_var_tur,dt,m,labels,replace) for d in dates]
        with Pool() as pool:
            pool.starmap(process_day,args)
    else:
        raise ValueError(f"{mode} is not a valid processing mode (must be serial or parallel)")
