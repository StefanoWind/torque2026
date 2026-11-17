# -*- coding: utf-8 -*-
"""
Plot data during a frontal passage
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import pyart
import glob
import matplotlib.gridspec as gridspec
import warnings
import pandas as pd
import utm
import rasterio
import requests
import truststore
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

# the NREL network re-signs TLS traffic, so certificates must be checked against the Windows store
truststore.inject_into_ssl()

#%% Inputs
source_msn=os.path.join(cd,'data','mesonet','*mdf')
source_msn_layout=os.path.join(cd,'data','mesonet','geoinfo.csv')
source_rad=os.path.join(cd,'data','nexrad','*ar2v')
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
sites_trp=['B','C1a','G']
wf_sel=['King Plains','Armadillo Flats','Breckinridge']

avg_time=10#[min] prebore period
max_range_rad=250#[km] max radar range plotted
R_eff=6371000*4/3#[m] effective Earth radius for beam propagation [Doviak and Zrnic, 1993]

dem_file=os.path.join(cd,'data','dem_bore_map.tif')
dem_url='https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
dem_epsg=32614#UTM zone 14N
dem_size=(900,800)#[pixels] DEM resolution (x,y)

make_video=False

#graphics
sel_plot=[0,8,16]
xlim=[475,700]#[km]
ylim=[3925,4125]#[km]
isoline_heights=[0.5,1,1.5,2,3]#[km] lowest-beam heights above the local ground shown as isolines

#%% Functions
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

def beam_height(s,elevation):
    #beam-center height above the antenna vs. ground distance s, 4/3 effective Earth radius model [Doviak and Zrnic, 1993]
    theta=np.radians(elevation)
    return R_eff*np.cos(theta)/np.cos(theta+s/R_eff)-R_eff

def read_lowest_tilt(file):
    radar=pyart.io.read_nexrad_archive(file).extract_sweeps([0])

    #filter bad data
    gatefilter=pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked("reflectivity")

    #gate-edge locations along the curved beam, mapped to UTM through lat/lon
    n_gates=np.sum(radar.range['data']<=max_range_rad*1000)
    x,y,_=radar.get_gate_x_y_z(0,edges=True)
    lon,lat=pyart.core.cartesian_to_geographic_aeqd(x[:,:n_gates+1],y[:,:n_gates+1],
                                                    radar.longitude['data'][0],radar.latitude['data'][0])
    x_utm,y_utm,_,_=utm.from_latlon(lat,lon,force_zone_number=14)

    Z=np.ma.masked_where(gatefilter.gate_excluded,radar.fields['reflectivity']['data'])[:,:n_gates]
    #mid-sweep time of the lowest tilt
    time_rad=np.datetime64(radar.time['units'][-20:-1])+np.timedelta64(int(np.round(np.mean(radar.time['data']))),'s')
    return x_utm/1000,y_utm/1000,Z,time_rad,float(radar.fixed_angle['data'][0])

def draw_isolines(ax):
    #isolines of the lowest-beam center height above the local ground
    cs=ax.contour(dem_x,dem_y,beam_agl/1000,levels=isoline_heights,colors='k',linestyles='-',linewidths=0.5,alpha=0.5)
    ax.clabel(cs,fmt='%g km',fontsize=9,inline=True)
    ax.plot(x_rad/1000,y_rad/1000,'^k',markersize=8)

#%% Initialiation

#read mesonet layout
msn_lyt=pd.read_csv(source_msn_layout).set_index('stid')
xy=utm.from_latlon(msn_lyt['nlat'].values, msn_lyt['elon'].values)
msn_lyt['x']=xy[0]
msn_lyt['y']=xy[1]

#read AWAKEN leyout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#load data
files_msn=sorted(glob.glob(source_msn))
files_rad=sorted(glob.glob(source_rad))

os.makedirs(os.path.join(cd,'figures','mesonet'),exist_ok=True)

#radar location and lowest-tilt elevation (from the first volume)
radar=pyart.io.read_nexrad_archive(files_rad[0])
lat_rad=float(radar.latitude['data'][0])
lon_rad=float(radar.longitude['data'][0])
alt_rad=float(radar.altitude['data'][0])#[m a.s.l.] antenna altitude
elevation_low=float(radar.fixed_angle['data'][0])
x_rad,y_rad,_,_=utm.from_latlon(lat_rad,lon_rad,force_zone_number=14)

dem_x,dem_y,dem=get_dem(dem_file,xlim,ylim,dem_size)

#%% Main

#read mesonet data
Data_msn=xr.Dataset()
for f in files_msn:
    df = pd.read_csv(f,delim_whitespace=True, skiprows=2,na_values=[-998, -999])
    datestr=str(os.path.basename(f))[:-4]
    time=np.datetime64(f'{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[8:10]}:{datestr[10:12]}:00')

    ds = xr.Dataset.from_dataframe(df.set_index("STID"))
    ds = ds.expand_dims(time=[time])

    if 'PRES' in Data_msn.data_vars:
        Data_msn=xr.concat([Data_msn, ds],dim='time')
    else:
        Data_msn=ds

#pressure perturbation
dp=(Data_msn.PRES/Data_msn.PRES.where(Data_msn.time<Data_msn.time[0]+np.timedelta64(avg_time,'m')).mean(dim='time')-1)*100

#reconcile station locations
x_msn=[]
y_msn=[]
for s in Data_msn.STID.values:
    x_msn=np.append(x_msn,msn_lyt['x'].loc[s])
    y_msn=np.append(y_msn,msn_lyt['y'].loc[s])

#AWAKEN turbines
Turbines['wind_plant']=Turbines.wind_plant.where(Turbines.wind_plant!='unknown Garfield County','Armadillo Flats')
x_tur=[]
y_tur=[]
for wf in wf_sel:
    x_tur=np.append(x_tur,Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values)
    y_tur=np.append(y_tur,Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values)

#lowest-beam altitude above local terrain (UTM distance used as ground distance, scale error <0.1%)
DEM_X,DEM_Y=np.meshgrid(dem_x*1000,dem_y*1000)
beam_agl=alt_rad+beam_height(np.hypot(DEM_X-x_rad,DEM_Y-y_rad),elevation_low)-dem

#%% Plots

#lowest-beam altitude above terrain
fig=plt.figure(figsize=(9,7))
ax=plt.gca()
cf=ax.contourf(dem_x,dem_y,beam_agl/1000,levels=np.arange(0,3.51,0.25),cmap='viridis',extend='both')
draw_isolines(ax)
ax.plot(x_tur/1000,y_tur/1000,'.w',markersize=1)
ax.set_aspect('equal')
plt.xlabel('W-E [km]')
plt.ylabel('S-N [km]')
plt.xlim(xlim)
plt.ylim(ylim)
plt.grid()
plt.title(f'KVNX {elevation_low:.2f}$^\\circ$ beam center above terrain')
plt.colorbar(cf,label='Beam altitude a.g.l. [km]')
plt.tight_layout()

#plot selected frames
fig=plt.figure(figsize=(18,5))
gs = gridspec.GridSpec(1, 5,width_ratios=[1,1,1,0.05,0.05])

ctr=0
for f in np.array(files_rad)[sel_plot]:
    ax=fig.add_subplot(gs[0,ctr])

    x_gate,y_gate,Z,time_rad,elevation=read_lowest_tilt(f)

    #interpolate pressure data
    dp_int=dp.interp(time=time_rad)

    #plot
    pc=plt.pcolormesh(x_gate,y_gate,Z,cmap='Spectral_r',vmin=-10,vmax=50)
    draw_isolines(ax)
    plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
    sc=plt.scatter(x_msn/1000,y_msn/1000,s=50,c=dp_int,cmap='seismic',vmin=-0.5,vmax=0.5,edgecolor='k',linewidth=2,zorder=10)

    ax.set_aspect('equal')
    plt.xlabel('W-E [km]')
    if ctr==0:
        plt.ylabel('S-N [km]')
    else:
        ax.set_yticklabels([])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(np.arange(475,701,25))
    plt.yticks(np.arange(3925,4126,25))
    plt.grid()
    plt.title(str(time_rad).replace('T',' ')+' UTC')
    ctr+=1

cax=fig.add_subplot(gs[0,-2])
cbar=plt.colorbar(pc,cax=cax)
cbar.ax.set_title(r'$Z$ [dBZ]  ', pad=10)
cax=fig.add_subplot(gs[0,-1])
cbar=plt.colorbar(sc,cax=cax)
cbar.ax.set_title(r'  $\Delta p$ [%]', pad=10)
cbar.ax.set_yticks([-0.5,0,0.5])

#plot all
if make_video:
    ctr=0
    for f in files_rad:
        plt.figure(figsize=(12,7))
        ax=plt.gca()
        x_gate,y_gate,Z,time_rad,elevation=read_lowest_tilt(f)

        #interpolate pressure data
        dp_int=dp.interp(time=time_rad)

        #plot
        pc=plt.pcolormesh(x_gate,y_gate,Z,cmap='Spectral_r',vmin=-10,vmax=50)
        draw_isolines(ax)
        plt.plot(x_tur/1000,y_tur/1000,'.k',markersize=1)
        sc=plt.scatter(x_msn/1000,y_msn/1000,s=50,c=dp_int,cmap='seismic',vmin=-0.5,vmax=0.5,edgecolor='k',linewidth=2,zorder=10)

        ax.set_aspect('equal')

        plt.xlabel('W-E [km]')
        plt.ylabel('S-N [km]')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        plt.title(str(time_rad).replace('T',' ')+' UTC')

        plt.colorbar(pc,label=r'$Z$ [dBZ]',location='right')
        plt.colorbar(sc,label=r'$\Delta p$ [%]',location='right')
        plt.tight_layout()
        plt.savefig(os.path.join(cd,'figures','mesonet',f'{ctr:02.0f}_mesonet.png'))
        plt.close()
        ctr+=1
