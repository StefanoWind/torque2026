# -*- coding: utf-8 -*-
"""
Identity dominant spatial wave in the SCADA
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib.dates as mdates
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','20230805.100000.20230805.115959.scada.nc')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
wf='King Plains'

t1=0#initial timestep
t2=7200#final timestep
skip=60#skip timesteps

lambdas=np.arange(1,25.1,1)#wavelength search space
wds=np.arange(180,359,5)#direction search space
dt_ma=60#[s]# averaging window

#QC
min_f=0#[m/s] min value
max_f=30#[m/s] max value

#graphics
sel_plot=[0,8,11,16]

#%% Functions
def lasso(x,y,f,k_search,theta_search,N_grid=100,margin=0):
    '''
    Identify dominant mode through LASSO
    '''
    from sklearn.linear_model import Lasso
    
    #make search grid
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
    
    #rmse of dominant mode
    rmse=np.sum((f-A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*x+np.sin(theta_dom)*y)+phi[i,j]))**2)**0.5
    
    return A, phi, k_dom, theta_dom, X, Y, f_dom, rmse

def bilinear(x, y, a, b, c):
    return a + b*x + c*y 

#%% Initialization
Data=xr.open_dataset(source)
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})
os.makedirs(os.path.join(cd,'figures','lasso'),exist_ok=True)

#farm layout
x=[]
y=[]
for tid in Data.turbine.values:
    x=np.append(x,Turbines.x_utm.where(Turbines.name==tid,drop=True).values/1000)
    y=np.append(y,Turbines.y_utm.where(Turbines.name==tid,drop=True).values/1000)

time_sel=np.arange(t1,t2,skip)

#zeroing
time_all=np.array([],dtype='datetime64')
A_all=[]
lambda_all=[]
wd_all=[]
rmse_all=[]

#%% Main

#rolling mean
dt=np.float64(np.mean(np.diff(Data.time.values)))/10**9
ws=Data.WindSpeed.rolling(time=int(dt_ma/dt), center=True).mean()

#apply LASSO
ctr=0
for i_t in time_sel:
   
    f=ws.isel(time=i_t).values
    sel=(f>min_f)*(f<max_f)
    
    if np.sum(sel)>0:
        
        #detrend
        A = np.column_stack([
            np.ones_like(x[sel]),   
            x[sel],                
            y[sel]]) 
            
        coeffs, *_ = np.linalg.lstsq(A, f[sel], rcond=None)
        a, b, c = coeffs
        
        df=f[sel]-bilinear(x[sel], y[sel], a, b, c)
        
        #LASSO
        A, phi, k_dom, theta_dom, X, Y, f_dom, rmse=lasso(x[sel],y[sel],df,2*np.pi/lambdas,np.radians(270-wds),margin=10)
        lambda_dom=2*np.pi/k_dom
        wd_dom=(270-np.degrees(theta_dom))%360
        
        #plot
        fig=plt.figure(figsize=(12,5))
        ax=plt.subplot(1,2,1)
        pc=plt.pcolor(X,Y,f_dom,vmin=-4,vmax=4,cmap='seismic')
        plt.scatter(x[sel],y[sel],s=20,c=df,vmin=-4,vmax=4,cmap='seismic',edgecolor='w')
        plt.xlabel('W-E [km]')
        plt.ylabel('S-N [km]')
        plt.xlim([628,649])
        plt.ylim([4023,4034])
        plt.xticks(np.arange(630,646,5))
        plt.yticks(np.arange(4025,4031,5))
        plt.grid()
        plt.title(str(Data.time.isel(time=i_t).values).replace('T',' ')[:-10])
        plt.colorbar(pc,label=r'$\tilde{U}_h$')
        
        ax = fig.add_subplot(1,2,2,projection='polar')
        pcm = ax.pcolormesh(np.radians(wds), lambdas, A, shading='auto', cmap='hot')
        plt.plot(np.radians(wd_dom),lambda_dom,'xg')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.title(r'$\lambda='+str(np.round(2*np.pi/k_dom,1))+r'$ km, $\theta='+str(np.round((270-np.degrees(theta_dom))%360))+'^\circ$')
        plt.colorbar(pcm,label="Amplitude")
        plt.show()
        
        plt.savefig(os.path.join(cd,'figures','lasso',f'{ctr:02.0f}'))
        plt.close()
       
        time_all=np.append(time_all,Data.time.isel(time=i_t).values)
        A_all=np.append(A_all,np.max(A))
        lambda_all=np.append(lambda_all,lambda_dom)
        wd_all=np.append(wd_all,wd_dom) 
        rmse_all=np.append(rmse_all,rmse)
        ctr+=1

#%% Plots
plt.figure(figsize=(18,6))
ax=plt.subplot(3,1,1)
plt.plot(time_all,A_all,'.-k')
plt.ylabel(r'Amplitude [m s${-1}$]',rotation=85)
plt.yticks([0,.5,1.0,1.5,2.0])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim([Data.time[0],Data.time[-1]])
ax.set_xticklabels([])
plt.grid()
ax=plt.subplot(3,1,2)
plt.plot(time_all,lambda_all,'.-k')
plt.yticks([0,5,10,15,20])
plt.ylabel(r'Wavelength [km]',rotation=85)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim([Data.time[0],Data.time[-1]])
ax.set_xticklabels([])
plt.grid()
ax=plt.subplot(3,1,3)
plt.quiver(time_all,np.zeros(len(time_all)),np.cos(np.radians(270-wd_all)),np.sin(np.radians(270-wd_all)),color='k',width=0.002)
plt.xlabel('Time (UTC)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlim([Data.time[0],Data.time[-1]])
ax.set_yticks([])
plt.grid()

