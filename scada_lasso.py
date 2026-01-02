# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:40:24 2026

@author: sletizia
"""

# -*- coding: utf-8 -*-
"""
Analyze SCADA data during bore
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import glob
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Lasso

import warnings
import utm
import pyart
import pandas as pd
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.00')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
wf='King Plains'
D=127
t1=2800
t2=3800
skip=60#[s]
power_rated=2800
z_rad=100
make_video=False
time_shift_rad=120#[s] shift radar time
lambdas=np.arange(-50,50.1,1)
dt_ma=60#[s]

#graphics
sel_plot=[0,8,11,16]

#%% Functions
def lasso(x,y,f,kx_search,ky_search,N_grid=100):
    from sklearn.linear_model import Lasso
    
    KX, KY = np.meshgrid(kx_search, ky_search, indexing="ij")
    N=len(f)
    Nk = KX.size
    
    #build modal matrix
    A_cos = np.zeros((N, Nk))
    A_sin = np.zeros((N, Nk))

    for i, (kx, ky) in enumerate(zip(KX.ravel(), KY.ravel())):
        phase = kx*x + ky*y
        A_cos[:, i] = np.cos(phase)
        A_sin[:, i] = np.sin(phase)

    A = np.hstack([A_cos, A_sin])
    
    #LASSO optimization
    lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=6000)
    lasso.fit(A, f)

    coef = lasso.coef_

    a = coef[:Nk]      # cosine coefficients
    b = coef[Nk:]      # sine coefficients
    
    #amplitude/phase
    A = np.sqrt(a**2 + b**2).reshape(KX.shape)
    phi = np.arctan2(-b, a).reshape(KX.shape)
    
    #dominant mode
    ij=np.argmax(A)
    i,j=np.unravel_index(ij, A.shape)
    
    X,Y=np.meshgrid(np.linspace(np.min(x),np.max(x),N_grid),np.linspace(np.min(y),np.max(y),N_grid),indexing='ij')
    f_dom=A[i,j]*np.cos(kx_search[i]*X+ky_search[j]*Y+phi[i,j])
    
    return A, phi, kx_search[i], ky_search[j], X, Y, f_dom
    
#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
Data=xr.Dataset()
os.makedirs(os.path.join(cd,'figures','lasso'),exist_ok=True)


#read scada
ctr=0
for file in os.listdir(source):
    scada_df=pd.read_csv(os.path.join(source,file))
    scada_df=scada_df.rename(columns={scada_df.columns[0]: "time"}).set_index('time')
    scada_df.index= pd.to_datetime(scada_df.index, utc=True).tz_convert(None)
    
    #extract turbine and variable list
    if ctr==0:
        _vars=[]
        turbines=[]
        for c in scada_df.columns:
            if 'Turbine' in c:
                # scada_df=scada_df.rename(columns={c: c.split('Turbine')[-1]})
                turbines=np.append(turbines,c.split('Turbine')[-1][:2])
                _vars=np.append(_vars,c.split('.')[-1])
    turbines=np.unique(turbines)
    _vars=np.unique(_vars)
    
    #build dataset
    scada_ds=xr.Dataset()
    for v in _vars:
        data=np.zeros((len(scada_df.index),len(turbines)))
        i_t=0
        for t in turbines:
            data[:,i_t]=scada_df[f'PKGP1HIST01.OKWF001_KP_Turbine{t}.{v}'].values
            i_t+=1
        scada_ds[v]=xr.DataArray(data,coords={'time':scada_df.index.values,'turbine':turbines})
    
    #stack
    if 'time' not in Data.coords:
        Data=scada_ds
    else:
        Data=xr.concat([Data,scada_ds],dim='time')
        
    ctr+=1

x=[]
y=[]
for tid in Data.turbine.values:
    x=np.append(x,Turbines.x_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000)
    y=np.append(y,Turbines.y_utm.where(Turbines.name==f'{tid[0]}0{tid[1]}',drop=True).values/1000)


lambdas=lambdas[np.abs(lambdas)>0.01]
kx_grid=2*np.pi/lambdas
ky_grid=2*np.pi/lambdas
KX, KY = np.meshgrid(kx_grid, ky_grid, indexing='ij')
X,Y=np.meshgrid(np.arange(0,30.1,0.1),np.arange(0,30.1,0.1), indexing='ij')
Nk = KX.size

#zeroing
A_all=[]
k_all=[]
theta_all=[]

#%% Main
ws=Data.WindSpeed.rolling(time=60, center=True).mean()
ctr=0
for i_t in np.arange(t1,t2+1,skip):
    f=ws.isel(time=i_t).values
    sel=f>0
    df=f[sel]-np.nanmean(f[sel])
    
    A, phi, kx_dom, ky_dom, X, Y, f_dom=lasso(x[sel],y[sel],df,2*np.pi/lambdas,2*np.pi/lambdas)
    
    plt.figure(figsize=(12,5))
    ax=plt.subplot(1,2,1)
    pc=plt.pcolor(X,Y,f_dom,vmin=np.min(df),vmax=np.max(df),cmap='seismic')
    plt.scatter(x[sel],y[sel],s=20,c=df,vmin=np.min(df),vmax=np.max(df),cmap='seismic',edgecolor='w')
    plt.xlabel('W-E [km]')
    plt.ylabel('S-N [km]')
    plt.xlim([628,649])
    plt.ylim([4023,4034])
    plt.xticks(np.arange(630,646,5))
    plt.yticks(np.arange(4025,4031,5))
    plt.grid()
    plt.title(str(Data.time.isel(time=i_t).values).replace('T',' ')[:-10])
    plt.colorbar(pc,label=r'$f$')
    
    ax=plt.subplot(1,2,2)
    plt.pcolor(lambdas, lambdas, A.T,cmap='hot')
    plt.plot(2*np.pi/kx_dom,2*np.pi/ky_dom,'xg',linewidth=3)
    plt.xlabel(r"$\lambda_x$")
    plt.ylabel(r"$\lambda_y$")
    plt.title(r'$\lambda_x='+str(np.round(2*np.pi/kx_dom,1))+'$ km,$\lambda_y='+str(np.round(2*np.pi/ky_dom,1))+'$ km')
    plt.colorbar(label="Amplitude")
    plt.show()
    
    plt.savefig(os.path.join(cd,'figures','lasso',f'{ctr:02.0f}'))
    plt.close()
    ctr+=1
    
    A_all.append(A_all,A)
    k_all=np.append()

