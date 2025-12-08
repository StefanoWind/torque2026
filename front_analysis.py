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
import matplotlib.dates as mdates
import glob
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
path_trp='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken'
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout
path_scada=os.path.join(cd,'data/scada')

sites_trp=['B','C1a','G']
sites_met=['B','C1a','G']
turb_sel=['A09','I02','G09']

sources_trp={'B':'sb.assist.tropoe.z01.c0/sb.assist.tropoe.z01.c0.20230805.001005.nc',
             'C1a':'sc1.assist.tropoe.z01.c0/sc1.assist.tropoe.z01.c0.20230805.001005.nc',
             'G':'sg.assist.tropoe.z01.c0/sg.assist.tropoe.z01.c0.20230805.001005.nc'}
sources_met={'A1':'sa1.met.z01.b0.20230805*nc',
             'A2':'sa2.met.z01.b0.20230805*nc',
             'A5':'sa5.met.z01.b0.20230805*nc',
             'A7':'sa7.met.z01.b0.20230805*nc',
             'B':'sb.met.z01.b0.20230805*nc',
             'C1a':'sc1.met.z01.b0.20230805*nc',
             'G':'sg.met.z01.b0.20230805*nc',}

path_trp='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken'
path_met=os.path.join(cd,'data','front')

times=np.array([np.datetime64('2023-08-05T10:40:00'),
       np.datetime64('2023-08-05T10:50:00'),
       np.datetime64('2023-08-05T11:00:00'),
       np.datetime64('2023-08-05T11:10:00'),
       np.datetime64('2023-08-05T11:20:00'),
       np.datetime64('2023-08-05T11:30:00'),
       np.datetime64('2023-08-05T11:40:00')])

farms_sel=['Armadillo Flats','King Plains','unknown Garfield County','Breckinridge']

height_assist=1#[m] height of the ASSIST a.g.l.

#QC
max_gamma=3 #maximumm gamma in TROPoe
max_rmsa=5 #max TROPoe error
min_lwp=5#[g/Kg] min LWP for clouds

#graphics
T_min=24
T_max=28
r_min=12
r_max=17
colors={'B':'g','C1a':'r','G':'b'}
turb_colors={'A09':'m','I02':'c','G09':'orange'}
site_sel='C1a'
turbine_files={'A09':'figures/Turbine_m.png',
               'I02':'figures/Turbine_c.png',
               'G09':'figures/Turbine_o.png'}

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

def draw_turbine(x,y,D,wd,turbine_file):
    import matplotlib.image as mpimg
    from matplotlib import transforms
    from matplotlib import pyplot as plt
    img = mpimg.imread(turbine_file)
    ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    tr = transforms.Affine2D().scale(D/1800,D/1800).translate(-20*D/700,-350*D/700).rotate_deg(90-wd).translate(x,y)
    ax.imshow(img, transform=tr + ax.transData)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
#%% Initialization

#read layout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#read TROPoe
T_trp={}
r_trp={}
for s in sites_trp:
    
    Data_trp=xr.open_dataset(os.path.join(path_trp,sources_trp[s]))
    
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<min_lwp).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=max_gamma
    qc_rmsa=Data_trp['rmsa']<=max_rmsa
    qc_cbh=Data_trp['height']<Data_trp['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data_trp['qc']=~qc+0
        
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
    #fix height
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
     
    T_trp[s]=Data_trp.temperature.where(Data_trp.qc==0)
    r_trp[s]=Data_trp.waterVapor.where(Data_trp.qc==0)
    
    Data_trp.close()

#read met
T_met={}
for s in sites_met:
    files=glob.glob(os.path.join(path_met,sources_met[s]))
    Data_met=xr.open_mfdataset(files)
    T_met[s]=Data_met.temperature.where(Data_met.qc_temperature==0)
    Data_met.close()
    
#read scada
Data_scd=xr.Dataset()
for file in os.listdir(path_scada):
    scada_df=pd.read_csv(os.path.join(path_scada,file))
    scada_df=scada_df.rename(columns={scada_df.columns[0]: "time"}).set_index('time')
    scada_df.index= pd.to_datetime(scada_df.index, utc=True).tz_convert(None)
    
    _vars=[]
    turbines=[]
    for c in scada_df.columns:
        if 'Turbine' in c:
            scada_df=scada_df.rename(columns={c: c.split('Turbine')[-1]})
            turbines=np.append(turbines,c.split('Turbine')[-1][:2])
            _vars=np.append(_vars,c.split('.')[-1])

    if 'time' not in Data_scd.coords:
        Data_scd=xr.Dataset.from_dataframe(scada_df)
    else:
        Data_scd=xr.concat([Data_scd,xr.Dataset.from_dataframe(scada_df)],dim='time')
    
#%% Main

#interpolate in time/height
T_trp_int={}
for s in sites_trp:
    T_trp_int[s]=T_trp[s].interp(height=2,time=times)
    
T_met_int={}
for s in sites_met:
    T_met_int[s]=T_met[s].interp(time=times)

#%% Plots
plt.close("all")
star_marker = MarkerStyle(three_point_star())
fig=plt.figure()
for wf in farms_sel:
    x_turbine=Turbines.x_utm.where(Turbines.wind_plant==wf,drop=True).values-Map.x_utm.sel(site='C1a').values
    y_turbine=Turbines.y_utm.where(Turbines.wind_plant==wf,drop=True).values-Map.y_utm.sel(site='C1a').values
    turbine_names=Turbines.name.where(Turbines.wind_plant==wf,drop=True)
    for xt,yt,tn in zip(x_turbine,y_turbine,turbine_names):
        if str(tn.values) in turb_sel and wf=='King Plains':
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=30, color=turb_colors[str(tn.values)])
        else:
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=10, color='k')

for s in sites_trp:
    sc=plt.scatter(Map.x_utm.sel(site=s)-Map.x_utm.sel(site='C1a'),Map.y_utm.sel(site=s)-Map.y_utm.sel(site='C1a'),
                c=colors[s],s=200,edgecolor='k',zorder=10)
    
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.xlim([-22000,15000])
plt.ylim([-16000,14000])
plt.grid()
plt.gca().set_aspect('equal')

#tropoe
matplotlib.rcParams['font.size'] = 16
fig=plt.figure(figsize=(18,4))
s=site_sel
cf=plt.contourf(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),cmap='hot',extend='both')
plt.contour(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),colors='k',linewidths=1,alpha=0.25,extend='both')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time (UTC)')
plt.ylabel(r'$z$ [m]')
plt.grid()
plt.xlim([T_met[s].time[0],T_met[s].time[-1]])
plt.ylim([-20,500])
plt.colorbar(cf,label=r'$T$ [$^\circ$C]',ticks=np.arange(T_min,T_max+0.1))

# ax=plt.subplot(2,1,2)
# cf=plt.contourf(r_trp[s].time,r_trp[s].height,r_trp[s].T,np.arange(r_min,r_max+.1,0.1),cmap='Blues',extend='both')
# plt.contour(r_trp[s].time,r_trp[s].height,r_trp[s].T,np.arange(r_min,r_max+.1,0.1),colors='k',linewidths=1,alpha=0.25,extend='both')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.xlabel('Time (UTC)')
# plt.ylabel(r'$z$ [m]')
# plt.grid()
# plt.xlim([T_met[s].time[0],T_met[s].time[-1]])
# plt.ylim([-20,500])
# plt.colorbar(cf,label=r'$r$ [g kg$^{-1}$]',ticks=np.arange(r_min,r_max+0.1))

#time history of T
plt.figure(figsize=(18,6))
for s in sites_met:
    plt.plot(T_met[s].time,T_met[s],color=colors[s],linewidth=2)
    plt.plot(T_trp[s].time,T_trp[s].isel(height=0),'^',color=colors[s],markersize=15,markeredgecolor='k')
plt.xlim([T_met[s].time[0],T_met[s].time[-1]])
plt.ylim([T_min,T_max])
plt.ylabel(r'$T$ [$^\circ$C]')
plt.xlabel('Time (UTC)')
plt.legend()
plt.grid()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

#scada
matplotlib.rcParams['font.size'] = 18
plt.figure(figsize=(18,10))
ax=plt.subplot(2,1,1)
time_sel=Data_scd.time.values[300:-1:600]
ctr=0
for t in turb_sel:
    for i in range(len(time_sel)):
        yaw=Data_scd[f'{t.replace("0","")}.Nacelle_Position'].sel(time=time_sel[i])
        draw_turbine(np.where(Data_scd.time==time_sel[i])[0][0],ctr,500,yaw,turbine_files[t])
    ctr+=500
plt.xlim([0,len(Data_scd.time)])
plt.ylim([-250,1250])
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(2,1,2)
for t in turb_sel:
    plt.plot(Data_scd.time,Data_scd[f'{t.replace("0","")}.ActivePower']/2800,color=turb_colors[t])
plt.ylabel('Normalized power')
plt.xlim([T_met[s].time[0],T_met[s].time[-1]])
plt.grid()
plt.xlabel('Time (UTC)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
