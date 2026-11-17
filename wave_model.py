# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:39:35 2026

@author: sletizia
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs

time=np.arange(0,1201,10)#[s] time grid of true field at the lidar location
z=np.arange(0,2001,10)
H=1700
lambda_x=6000
l1=0.0015
A=4
c=20
phase=np.pi/2

#virtual lidar (alpha is beam angle from the x axis, i.e. the wave propagation direction)
x_lidar=10000#[m] lidar location
ele_lidar=np.array([45,45,45,45,45,90])#[deg] beam elevation
alpha_lidar=np.array([0,72,144,216,288,0])#[deg] beam angle
dt_beam=4#[s] time per beam
T_lidar=1200#[s] duration of lidar simulation

#%% Functions
def wave(x,z,t,phase):
    Z=z-H
    u=np.where(Z<0,A/k*(m2*np.cos(m1*Z)+m1*np.sin(m1*Z)),A*m2/k*np.exp(-m2*Z))*np.sin(k*(x-c*t)+phase)
    w=np.where(Z<0,A*(np.cos(m1*Z)-m2/m1*np.sin(m1*Z)),A*np.exp(-m2*Z))*np.cos(k*(x-c*t)+phase)
    return u,w

#%% Initialization

T,Z=np.meshgrid(time,z-H,indexing='ij')

k=2*np.pi/lambda_x

m1=(l1**2-k**2)**0.5
m2=-m1/np.tan(m1*H)

print(f'Min l1 = {(np.pi**2/(4*H**2)+k**2)**0.5} 1/m')
print(f'Max l1 = {(np.pi**2/(H**2)+k**2)**0.5} 1/m')
assert (m1*H>np.pi/2)*(m1*H<np.pi), "Condition violated"

#%% Main
u,w=wave(x_lidar,Z+H,T,phase)

#virtual lidar sampling: each beam probes the field at its own time and horizontal offset (field is uniform in y)
Nb=len(ele_lidar)
t_beam=np.arange(0,T_lidar,dt_beam)
Nt=len(t_beam)
ele_b=ele_lidar[np.arange(Nt)%Nb]
alpha_b=alpha_lidar[np.arange(Nt)%Nb]
x_b=x_lidar+z[:,None]/np.tan(np.radians(ele_b))*np.cos(np.radians(alpha_b))
u_b,w_b=wave(x_b,z[:,None],t_beam,phase)
rws=u_b*np.cos(np.radians(ele_b))*np.cos(np.radians(alpha_b))+w_b*np.sin(np.radians(ele_b))

#reconstruction by least squares over a sliding window of Nb consecutive beams (as in Process_ground_lidar.py)
A_inv={}
for i in range(Nb):
    A_b=np.array([[np.cos(np.radians(b))*np.cos(np.radians(a)),np.cos(np.radians(b))*np.sin(np.radians(a)),np.sin(np.radians(b))]
                  for a,b in zip(alpha_b[i:i+Nb],ele_b[i:i+Nb])])
    A_inv[i]=np.linalg.pinv(A_b)

u_lidar=[]
w_lidar=[]
t_lidar=[]
for i in range(Nt-Nb+1):
    vel_vector=A_inv[i%Nb]@rws[:,i:i+Nb].T
    u_lidar.append(vel_vector[0,:])
    w_lidar.append(vel_vector[2,:])
    t_lidar.append(np.mean(t_beam[i:i+Nb]))
u_lidar=np.array(u_lidar)
w_lidar=np.array(w_lidar)
t_lidar=np.array(t_lidar)

#%% Plots
fig,axs=plt.subplots(2,2,sharex=True,sharey=True,figsize=(16,8))
plt.sca(axs[0,0])
plt.pcolor(T,Z+H,u,cmap='seismic',vmin=-A*1.25,vmax=A*1.25)
plt.colorbar(label=r'$u$ [m s$^{-1}$]')
plt.ylabel(r'$z$ [m]')
plt.title('True')

plt.sca(axs[1,0])
plt.pcolor(T,Z+H,w,cmap='seismic',vmin=-A*1.25,vmax=A*1.25)
plt.colorbar(label=r'$w$ [m s$^{-1}$]')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$z$ [m]')

plt.sca(axs[0,1])
plt.pcolor(t_lidar,z,u_lidar.T,cmap='seismic',vmin=-A*1.25,vmax=A*1.25)
plt.colorbar(label=r'$u$ [m s$^{-1}$]')
plt.title('Lidar')

plt.sca(axs[1,1])
plt.pcolor(t_lidar,z,w_lidar.T,cmap='seismic',vmin=-A*1.25,vmax=A*1.25)
plt.colorbar(label=r'$w$ [m s$^{-1}$]')
plt.xlabel(r'$t$ [s]')
plt.xlim([t_lidar[0],t_lidar[-1]])
plt.tight_layout()


