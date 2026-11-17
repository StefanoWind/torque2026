# -*- coding: utf-8 -*-
"""
Plot layout
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
import utm
import xarray as xr
import rasterio
import requests
import truststore
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

# the NREL network re-signs TLS traffic, so certificates must be checked against the Windows store
truststore.inject_into_ssl()

#%% Inputs
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
source_mesonet=os.path.join(cd,'data','mesonet','geoinfo.csv')
dem_file=os.path.join(cd,'data','dem_awaken.tif')
dem_file_large=os.path.join(cd,'data','dem_awaken_large.tif')
dem_url='https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
dem_epsg=32614#UTM zone 14N, same as layout x_utm/y_utm
dem_size=(800,550)#[pixels] DEM resolution (x,y)
xlim=[610,650]#[km]
ylim=[4010,4037.5]#[km]
dem_size_large=(800,650)#[pixels] DEM resolution of the large map (x,y)
xlim_large=[540,720]#[km]
ylim_large=[3960,4090]#[km]
ele_min=200
ele_max=500
radar_latlon=(36.740617,-98.127717)#KVNX
date_sel=20230805#selects Mesonet stations active on this date
farms_sel=['Armadillo Flats','King Plains','unknown Garfield County','Breckinridge']
sites_trp=['B','C1a','G']
sites_met=['A1']
sites_ws=['H']
turbines_sel=['E06']
colors={'A09':'g','I02':'b','G09':'r','E06':'m'}

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

def get_dem(file,xlim,ylim,size):
    # DEM over the map extent (USGS 3DEP) in UTM, cached locally and re-downloaded if the extent changed
    cached=False
    if os.path.isfile(file):
        with rasterio.open(file) as src:
            b=src.bounds
        # the server pads the extent to keep square pixels, so the cache only needs to cover the requested extent
        cached=b.left<=xlim[0]*1000 and b.right>=xlim[1]*1000 and b.bottom<=ylim[0]*1000 and b.top>=ylim[1]*1000
    if not cached:
        r=requests.get(dem_url,params=dict(bbox=f'{xlim[0]*1000},{ylim[0]*1000},{xlim[1]*1000},{ylim[1]*1000}',bboxSR=dem_epsg,imageSR=dem_epsg,
                                           size=f'{size[0]},{size[1]}',format='tiff',pixelType='F32',f='image'),timeout=120)
        r.raise_for_status()
        with open(file,'wb') as f:
            f.write(r.content)
    with rasterio.open(file) as src:
        dem=src.read(1)
        b=src.bounds
    return np.linspace(b.left,b.right,dem.shape[1])/1000,np.linspace(b.top,b.bottom,dem.shape[0])/1000,dem

#%% Initialization
#read layout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

dem_x,dem_y,dem=get_dem(dem_file,xlim,ylim,dem_size)
dem_x_large,dem_y_large,dem_large=get_dem(dem_file_large,xlim_large,ylim_large,dem_size_large)

#read Mesonet stations active on the selected date
Mesonet=pd.read_csv(source_mesonet)
Mesonet=Mesonet[(Mesonet.datc<=date_sel)&(Mesonet.datd>=date_sel)]
x_mes,y_mes,_,_=utm.from_latlon(Mesonet.nlat.values,Mesonet.elon.values,force_zone_number=14)
Mesonet['x']=x_mes/1000
Mesonet['y']=y_mes/1000
Mesonet=Mesonet[Mesonet.x.between(*xlim_large)&Mesonet.y.between(*ylim_large)]

x_radar,y_radar,_,_=utm.from_latlon(*radar_latlon,force_zone_number=14)

#%% Plots
plt.close("all")
matplotlib.rcParams['font.size'] = 14
star_marker = MarkerStyle(three_point_star())
fig=plt.figure(figsize=(9,7))
ax=plt.gca()
cf=ax.contourf(dem_x,dem_y,dem,levels=np.arange(ele_min,ele_max+1,1),cmap='terrain',vmin=dem.min()-30)
cs=ax.contour(dem_x,dem_y,dem,levels=np.arange((np.ceil(ele_min)/10)*10,ele_max+1,10),colors='k',linewidths=0.4,alpha=0.5)
ax.clabel(cs,fmt='%d',fontsize=8,inline=True)
for wf in farms_sel:
    x_turbine=Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
    y_turbine=Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values/1000
    turbine_names=Turbines.name.where(Turbines.wind_plant==wf,drop=True)
    for xt,yt,tn in zip(x_turbine,y_turbine,turbine_names):
        if str(tn.values) in turbines_sel and wf=='King Plains':
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=20, color=colors[str(tn.values)],label='Turbine loads',zorder=10)
        else:
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=10, color='k')

for s in sites_ws:
    sc=plt.scatter(Map.x_utm.sel(site=s)/1000,Map.y_utm.sel(site=s)/1000,
                c='c',s=100,edgecolor='k',zorder=10,marker='s',label='Wind profiling')
for s in sites_trp:
    sc=plt.scatter(Map.x_utm.sel(site=s)/1000,Map.y_utm.sel(site=s)/1000,
                c='r',s=100,edgecolor='k',zorder=10,marker='s',label='Thermodynamic profiling + met station')
    
for s in sites_met:
    sc=plt.scatter(Map.x_utm.sel(site=s)/1000,Map.y_utm.sel(site=s)/1000,
                c='w',s=100,edgecolor='k',zorder=10,marker='s',label='Met station')

plt.xlim(xlim)
plt.ylim(ylim)
ax.set_xticklabels([])
ax.set_yticklabels([])
handles,labels=ax.get_legend_handles_labels()
unique=dict(zip(labels,handles))
plt.legend(unique.values(),unique.keys(),draggable='True')
ax.set_aspect('equal')

#large map
matplotlib.rcParams['font.size'] = 12
fig=plt.figure(figsize=(9,7))
ax=plt.gca()
cf=ax.contourf(dem_x_large,dem_y_large,dem_large,levels=np.arange(ele_min,ele_max+1,1),cmap='terrain',vmin=dem_large.min()-30)
cs=ax.contour(dem_x_large,dem_y_large,dem_large,levels=np.arange(np.ceil(ele_min/50)*50,ele_max+1,50),colors='k',linewidths=0.4,alpha=0.5)
ax.clabel(cs,fmt='%d',fontsize=8,inline=True)
plt.scatter(Mesonet.x,Mesonet.y,c=(0.5,0.5,0.5),s=50,edgecolor='k',zorder=10,marker='s',label='Oklahoma Mesonet')
for _,row in Mesonet.iterrows():
    ax.annotate(row.stid,(row.x,row.y),xytext=(5,4),textcoords='offset points',fontsize=8,zorder=10,
                bbox=dict(facecolor='w',alpha=0.7,edgecolor='none',pad=1))
plt.scatter(x_radar/1000,y_radar/1000,c='orange',s=150,edgecolor='k',zorder=10,marker='s',label='NEXRAD KVNX')
ax.add_patch(Rectangle((xlim[0],ylim[0]),xlim[1]-xlim[0],ylim[1]-ylim[0],facecolor='none',edgecolor='k',linewidth=1.5,zorder=9,label='AWAKEN area'))
plt.xlabel('W-E [km]')
plt.ylabel('S-N [km]')
plt.xlim(xlim_large)
plt.ylim(ylim_large)
plt.legend(draggable='True')
ax.set_aspect('equal')
cax=make_axes_locatable(ax).append_axes('right',size='4%',pad=0.1)
plt.colorbar(cf,cax=cax,label='Elevation a.s.l. [m]',ticks=np.arange(np.ceil(ele_min/50)*50,ele_max+1,50))
