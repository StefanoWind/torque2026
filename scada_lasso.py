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

import warnings
import utm
import pyart
import pandas as pd
# warnings.filterwarnings('ignore')
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
t1=0
t2=7200
skip=60#[s]
power_rated=2800

lambdas=np.arange(1,25.1,1)
wds=np.arange(0,181,5)
dt_ma=60#[s]

#graphics
sel_plot=[0,8,11,16]

#%% Functions
def lasso(x,y,f,k_search,theta_search,N_grid=100,margin=0):
    from sklearn.linear_model import Lasso
    
    K, THETA = np.meshgrid(k_search, theta_search, indexing="ij")
    N=len(f)
    Nk = K.size
    
    #build modal matrix
    M_cos = np.zeros((N, Nk))
    M_sin = np.zeros((N, Nk))

    for i, (k, theta) in enumerate(zip(K.ravel(), THETA.ravel())):
        phase = k*(np.cos(theta)*x + np.sin(theta)*y)
        M_cos[:, i] = np.cos(phase)
        M_sin[:, i] = np.sin(phase)

    M = np.hstack([M_cos, M_sin])
    
    #LASSO optimization
    lasso = Lasso(alpha=0.1, fit_intercept=False, max_iter=6000)
    lasso.fit(M, f)

    coef = lasso.coef_

    a = coef[:Nk]      # cosine coefficients
    b = coef[Nk:]      # sine coefficients
    
    #amplitude/phase
    A = np.sqrt(a**2 + b**2).reshape(K.shape)
    phi = np.arctan2(-b, a).reshape(K.shape)
    
    #dominant mode
    ij=np.argmax(A)
    i,j=np.unravel_index(ij, A.shape)
    
    k_dom=k_search[i]
    theta_dom=theta_search[j]
    X,Y=np.meshgrid(np.linspace(np.min(x)-margin,np.max(x)+margin,N_grid),np.linspace(np.min(y)-margin,np.max(y)+margin,N_grid),indexing='ij')
    f_dom=A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*X+np.sin(theta_dom)*Y)+phi[i,j])
    
    rmse=np.sum((f-A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*x+np.sin(theta_dom)*y)+phi[i,j]))**2)**0.5
    
    return A, phi,k_dom,theta_dom, X, Y, f_dom,rmse

def bilinear(x, y, a, b, c):
    return a + b*x + c*y 

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

time_sel=np.arange(t1,t2,skip)

#zeroing
time_all=np.array([],dtype='datetime64')
A_all=[]
lambda_all=[]
wd_all=[]
rmse_all=[]

#%% Main
ws=Data.WindSpeed.rolling(time=60, center=True).mean()
ctr=0
for i_t in time_sel:
   
    f=ws.isel(time=i_t).values
    sel=f>0
    if np.sum(sel)>0:
        
        A = np.column_stack([
            np.ones_like(x[sel]),   
            x[sel],                
            y[sel]]) 
            
        coeffs, *_ = np.linalg.lstsq(A, f[sel], rcond=None)
        a, b, c = coeffs
        
        df=f[sel]-bilinear(x[sel], y[sel], a, b, c)

        A, phi, k_dom, theta_dom, X, Y, f_dom, rmse=lasso(x[sel],y[sel],df,2*np.pi/lambdas,np.radians(270-wds),margin=10)
        
        fig=plt.figure(figsize=(12,5))
        ax=plt.subplot(1,2,1)
        pc=plt.pcolor(X,Y,f_dom,vmin=-np.percentile(np.abs(df),90),vmax=np.percentile(np.abs(df),90),cmap='seismic')
        plt.scatter(x[sel],y[sel],s=20,c=df,vmin=-np.percentile(np.abs(df),90),vmax=np.percentile(np.abs(df),90),cmap='seismic',edgecolor='w')
        plt.xlabel('W-E [km]')
        plt.ylabel('S-N [km]')
        plt.xlim([628,649])
        plt.ylim([4023,4034])
        plt.xticks(np.arange(630,646,5))
        plt.yticks(np.arange(4025,4031,5))
        plt.grid()
        plt.title(str(Data.time.isel(time=i_t).values).replace('T',' ')[:-10])
        plt.colorbar(pc,label=r'$f$')
        
        ax = fig.add_subplot(1,2,2,projection='polar')
        pcm = ax.pcolormesh(np.radians(90-wds), lambdas, A, cmap='hot')
        plt.plot(theta_dom+np.pi,2*np.pi/k_dom,'xg')
        plt.title(r'$\lambda='+str(np.round(2*np.pi/k_dom,1))+r'$ km, $\theta='+str(np.round((270-np.degrees(theta_dom))%360))+'^\circ$')
        plt.colorbar(pcm,label="Amplitude")
        plt.show()
        
        plt.savefig(os.path.join(cd,'figures','lasso',f'{ctr:02.0f}'))
        plt.close()
       
        time_all=np.append(time_all,Data.time.isel(time=i_t).values)
        A_all=np.append(A_all,np.max(A))
        lambda_all=np.append(lambda_all,2*np.pi/k_dom)
        wd_all=np.append(wd_all,(270-np.degrees(theta_dom))%360) 
        rmse_all=np.append(rmse_all,rmse)
        ctr+=1

#%% Plots
plt.figure(figsize=(18,10))
ax=plt.subplot(3,1,1)
plt.plot(time_all,A_all,'k')
plt.ylabel(r'$A$ [m s${-1}$]')
ax.set_xticklabels([])
plt.grid()
ax=plt.subplot(3,1,2)
plt.plot(time_all,lambda_all,'k')
plt.ylabel(r'$\lambda$ [km]')
ax.set_xticklabels([])
plt.grid()
ax=plt.subplot(3,1,3)
plt.quiver(time_all,np.zeros(len(time_all)),np.cos(np.radians(90-wd_all)),np.sin(np.radians(90-wd_all)),color='k',width=0.002)
plt.xlabel('Time (UTC)')
ax.set_yticks([])
plt.grid()

