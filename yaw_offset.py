# -*- coding: utf-8 -*-
"""
10-min statistics (mean, std, min, max, valid-sample count) of the yaw setpoint offset WYAW.YwSptOff from King Plains 1-s SCADA.
"""
import os
cd=os.path.dirname(__file__)
import glob
import numpy as np
import xarray as xr

#%% Inputs
source=os.path.join(cd,'data/awaken/kp.turbine.z04.b0')
var='WYAW.YwSptOff'
dt=600#[s] averaging window
max_files=None#None to process all files

#%% Initialization
files=sorted(glob.glob(os.path.join(source,'*.nc')))[:max_files]
turbine_ids=np.array([os.path.basename(f).split('_WT')[1][:3] for f in files])
save_dir=source.replace('b0','c1')
os.makedirs(save_dir,exist_ok=True)

#%% Main
for turbine_id in np.unique(turbine_ids):
    Data=xr.open_mfdataset(np.array(files)[turbine_ids==turbine_id])[var].rename({'WTUR.DateTime':'time'}).compute()

    #bins labeled by their center
    Data_res=Data.resample(time=f'{dt}s')
    Stats=xr.Dataset({'avg':Data_res.mean(),
                      'std':Data_res.std(),
                      'min':Data_res.min(),
                      'max':Data_res.max(),
                      'num':Data_res.count()})
    Stats['time']=Stats.time+np.timedelta64(dt//2,'s')
    Stats=Stats.isel(time=Stats['num'].values>0)
    Stats.attrs={'variable':var,'window [s]':dt,'source':os.path.basename(source)}

    save_name=os.path.join(save_dir,f'YwSptOff_wt{turbine_id}.nc')
    Stats.to_netcdf(save_name)
    print(f'Saved {save_name}',flush=True)
