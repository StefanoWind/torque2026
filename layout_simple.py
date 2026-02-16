# -*- coding: utf-8 -*-
"""
Plot simple layout
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
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%%Inputs
source=os.path.join(cd,'data','awaken','kp.turbine.z02.00')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
turbines_sel=['E06']
wf='King Plains'

colors={'E06':'m'}

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
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#%% Plots
star_marker = MarkerStyle(three_point_star())
plt.figure()
turbines_wf=Turbines.name.where(Turbines.wind_plant==wf,drop=True).values
for t in turbines_wf:
    x_tur=Turbines.x_utm.where(Turbines.name==t,drop=True).values/1000
    y_tur=Turbines.y_utm.where(Turbines.name==t,drop=True).values/1000
    if t in turbines_sel and wf=='King Plains':
        plt.plot(x_tur,y_tur,'xk', marker=star_marker, markersize=30, color=colors[t],zorder=10)
    else:
        plt.plot(x_tur,y_tur,'xk', marker=star_marker, markersize=10, color='k')
        
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.xlim([625,650])
plt.ylim([4022,4035])
plt.gca().set_aspect('equal')
