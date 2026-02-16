# -*- coding: utf-8 -*-
"""
Plot layout
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
farms_sel=['Armadillo Flats','King Plains','unknown Garfield County','Breckinridge']
sites_trp=['B','C1a','G']
sites_ws=['H']
turb_sel=[]

#%% Functions
def three_point_star():
    # Points of a 3-pointed star (scaled and centered)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)

#%% Initialization
#read layout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})


#%% Plots
plt.close("all")
star_marker = MarkerStyle(three_point_star())
fig=plt.figure()
for wf in farms_sel:
    x_turbine=Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
    y_turbine=Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
    turbine_names=Turbines.name.where(Turbines.wind_plant==wf,drop=True)
    for xt,yt,tn in zip(x_turbine,y_turbine,turbine_names):
        if str(tn.values) in turb_sel and wf=='King Plains':
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=30, color='m',zorder=10)
        else:
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=10, color='k')

for s in sites_ws:
    sc=plt.scatter(Map.x_utm.sel(site=s)/1000,Map.y_utm.sel(site=s)/1000,
                c='c',s=100,edgecolor='k',zorder=10,marker='s')
for s in sites_trp:
    sc=plt.scatter(Map.x_utm.sel(site=s)/1000,Map.y_utm.sel(site=s)/1000,
                c='orange',s=100,edgecolor='k',zorder=10,marker='s')
    
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.xlim([610,650])
plt.ylim([4010,4037.5])
plt.gca().set_aspect('equal')