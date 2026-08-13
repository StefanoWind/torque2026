# -*- coding: utf-8 -*-
"""
Batch load statistics: computes per-10-min mean, maximum, valid-sample count and DEL of turbine load channels, run per day over a date range.

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
from matplotlib import pyplot as plt
from openfast_toolbox.tools.fatigue import equivalent_load
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
    sdate='2023-08-05T00:00:00'#start date
    edate='2023-08-06T00:00:00'#end date
    replace=True
    mode='serial'#serial or parallel
else:
    source=sys.argv[1]
    turbine_id=sys.argv[2]
    sdate=sys.argv[3]
    edate=sys.argv[4]
    replace=sys.argv[5]=='True'
    mode=sys.argv[6]#serial or parallel

loads_var=['tb_bend_foreaft','tb_bend_sideside','tb_bend_resultant','b1_bend_flap_root','b1_bend_edge_root','b1_bend_root_resultant']

wholer_exp={'tb_bend_foreaft':4,   'tb_bend_sideside':4,  'tb_bend_resultant':4,
            'b1_bend_flap_root':10,'b1_bend_edge_root':10,'b1_bend_root_resultant':10}#Wohler exponent
dt=600#[s] time step

#graphics
labels={'tb_bend_foreaft':r'$M_{tb,fa}$ [kNm]',
        'tb_bend_sideside':r'$M_{tb,ss}$ [kNm]',
        'tb_bend_resultant':r'$M_{tb,res}$ [kNm]',
        'b1_bend_root_resultant':r'$M_{br,res}$ [kNm]',
        'b1_bend_flap_root':r'$M_{br,fl}$ [kNm]',
        'b1_bend_edge_root':r'$M_{br,ed}$ [kNm]'}

loads_var_tur=[f'{turbine_id}_{v}' for v in loads_var]

#%% Functions
def process_day(d,source,turbine_id,loads_var,loads_var_tur,dt,labels,replace):

    #full-day, fixed-width time bins (bin centers used as the output time coordinate)
    bins_time=np.arange(d,
                        d+np.timedelta64(1,'D')+np.timedelta64(1,'s'),
                        np.timedelta64(dt,'s'))
    time_avg=bins_time[:-1]+(bins_time[1:]-bins_time[:-1])/2

    #find files
    d_str=str(d).split('T')[0].replace('-','')
    files=np.array(sorted(glob.glob(os.path.join(source,'*'+d_str+'*'))))

    save_name=os.path.join(source.replace('b0','c1'),os.path.basename(source).replace('b0','c1')+'.'+d_str+'.nc')

    if os.path.isfile(save_name)==False or replace:
        if len(files)>0:
            turbine_ids=np.array([f.split('.')[-2] for f in files])
            Stats=xr.Dataset()

            #read all loads
            try:
                Data=xr.open_mfdataset(files[turbine_ids==turbine_id])
                Data_qc=xr.Dataset()

                for v in loads_var_tur:
                    if v in Data.data_vars:
                        _num=[]
                        _avg=[]
                        _max=[]
                        _del=[]
                        
                        Data_qc[v]=Data[v].where(Data['qc_'+v]==0).compute()
                        for t1,t2 in zip(bins_time[:-1],bins_time[1:]):
                            Data_sel=Data_qc.where((Data_qc.time>=t1)*(Data_qc.time<t2),drop=True)#select time bin (half-open, avoids dropping samples on bin edges)

                            if len(Data_sel.time)>0:

                                L=Data_sel[v].values

                                #count
                                _num=np.append(_num,np.sum(~np.isnan(L)))

                                #avg
                                _avg=np.append(_avg,np.nanmean(L))

                                #max
                                _max=np.append(_max,np.nanmax(L))

                                #DEL
                                time=(Data_sel.time.values-Data_sel.time.values[0])/np.timedelta64(1,'s')#time in seconds
                                _del =np.append(_del, equivalent_load(time, L, m=wholer_exp[v[3:]]))#calculcate del

                                print(f'Calculated stats of {v} at {str(t1).replace("T"," ")}',flush=True)
                            else:
                                _del =np.append(_del, np.nan)
                                _avg =np.append(_avg, np.nan)
                                _max =np.append(_max, np.nan)
                                _num=np.append(_num,0)

                        #store data
                        Stats[v+'_num']=xr.DataArray(_num,coords={'time':time_avg})
                        Stats[v+'_avg']=xr.DataArray(_avg,coords={'time':time_avg})
                        Stats[v+'_max']=xr.DataArray(_max,coords={'time':time_avg})
                        Stats[v+'_del']=xr.DataArray(_del,coords={'time':time_avg})

                plt.figure(figsize=(18,10))
                ctr=1
                for v in loads_var_tur:
                    if v in Data.data_vars:
                        ax=plt.subplot(len(loads_var),1,ctr)
                        
                        plt.plot(Stats.time,Stats[f'{v}_avg'],'.-k',label='Avg')
                        plt.plot(Stats.time,Stats[f'{v}_max'],'.-r',label='Max')
                        plt.plot(Stats.time,Stats[f'{v}_del'],'.-b',label='DEL')
                        ax.set_xticklabels([])
                        plt.ylabel(labels[v[3:]])
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        plt.grid()
                        if ctr==1:
                            plt.legend()
                        ctr+=1

                #output
                plt.xlabel('Time (UTC)')
                plt.tight_layout()
                plt.savefig(os.path.join(source.replace('b0','c1'),os.path.basename(source).replace('b0','c1')+'.'+d_str+'.png'))
                plt.close()

                Stats.to_netcdf(save_name)
                print(f'Saved {os.path.basename(save_name)}',flush=True)

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
    os.makedirs(os.path.join(source.replace('b0','c1')),exist_ok=True)

    if mode=='serial':
        for d in dates:
            process_day(d,source,turbine_id,loads_var,loads_var_tur,dt,labels,replace)
    elif mode=='parallel':
        args=[(d,source,turbine_id,loads_var,loads_var_tur,dt,labels,replace) for d in dates]
        with Pool() as pool:
            pool.starmap(process_day,args)
    else:
        raise ValueError(f"{mode} is not a valid processing mode (must be serial or parallel)")
