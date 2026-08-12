# -*- coding: utf-8 -*-
"""
Scatterplot of turbine loads vs. wind speed with bin-averaged statistics
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
import xarray as xr
import glob
import warnings
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_loads=os.path.join(cd,'data/awaken/kp.turbine.z03.del/*nc')
source_scada=os.path.join(cd,'data/awaken/kp.turbine.z02.c0')
turbine_id='e6'
turbine_id_scada='wt041'#SCADA turbine id corresponding to turbine_id (Alias E06)
loads_var=['tb_bend_resultant','b1_bend_root_resultant']
ws_var='WMET.HorWdSpd_10m_Avg'
ws_flag_var='WS_Flag_10m_Percent'
pow_var='WTUR.W_10m_Avg'
pow_flag_var='PWR_Flag_10m_Percent'
min_power=2800*0.05
min_data_avail=0.5

bin_width=0.5#[m/s] wind speed bin width
max_ci_frac=0.1#max width of the bootstrap CI, as a fraction of the load's std

highlight_period=[np.datetime64('2023-08-05T10:00:00'),np.datetime64('2023-08-05T12:00:00')]#datetimes highlighted in the scatter

labels={'tb_bend_resultant':'DEL of tower-base bending moment [kNm]',
        'b1_bend_root_resultant':'DEL of blade-root bending moment [kNm]'}

#%% Functions
def bootstrap(x,M):
    '''
    Bootstrap sample drawer
    '''
    i=np.random.randint(0,len(x),size=(M,len(x)))
    x_BS=x[i]
    return x_BS

def filt_stat(x,func,perc_lim=[5,95]):
    '''
    Statistic with percentile filter
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
       
    return func(x_filt)

def filt_BS_stat(x,func,p_value=5,M_BS=100,min_N=10,perc_lim=[5,95]):
    '''
    Statstics with percentile filter and bootstrap
    '''
    x_filt=x.copy()
    lb=np.nanpercentile(x_filt,perc_lim[0])
    ub=np.nanpercentile(x_filt,perc_lim[1])
    x_filt=x_filt[(x_filt>=lb)*(x_filt<=ub)]
    
    if len(x_filt)>=min_N:
        x_BS=bootstrap(x_filt,M_BS)
        stat=func(x_BS,axis=1)
        BS=np.nanpercentile(stat,p_value)
    else:
        BS=np.nan
    return BS

def binned_avg_1d(x,f,bins,max_ci,perc_lim=[5,95],p_value=0.05):
    
    '''
    2D binned average with bootstrap convergence guard
    '''
    f_avg= stats.binned_statistic(x,f,statistic=lambda x:    filt_stat(x,np.nanmean,perc_lim=perc_lim),                          bins=bins)[0]
    f_low= stats.binned_statistic(x,f,statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=    p_value/2*100),bins=bins)[0]
    f_top= stats.binned_statistic(x,f,statistic=lambda x: filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bins)[0]
    
    f_qc=f_avg.copy()
    f_qc[f_top-f_low>max_ci]=np.nan
    f_qc[np.isnan(f_top-f_low)]=np.nan
    return f_qc

#%% Load data
files=sorted(glob.glob(source_loads))
Loads=xr.open_mfdataset(files).compute()

#restrict SCADA files to the days covered by the loads dataset
dates=pd.date_range(pd.Timestamp(Loads.time.values.min()).floor('D'),
                     pd.Timestamp(Loads.time.values.max()).floor('D'))
files_scada=[]
for d in dates:
    files_scada+=glob.glob(os.path.join(source_scada,f'*{d.strftime("%Y%m%d")}*{turbine_id_scada}*nc'))
Scada=xr.open_mfdataset(sorted(files_scada)).compute()
Scada=Scada.rename({'WTUR.DateTime':'time'})
Scada['time']=Scada.time+pd.Timedelta(minutes=5)#SCADA timestamps mark bin start, loads use bin centers


power=Scada[pow_var].where(Scada[pow_flag_var]>=min_data_avail)#kept available for future use
ws=Scada[ws_var].where(Scada[ws_flag_var]>=min_data_avail).where(power>min_power)

#align wind speed onto the load time grid
ws_aligned=ws.reindex(time=Loads.time.values).values

ws_bins=np.arange(0,np.nanmax(ws_aligned)+bin_width,bin_width)
ws_avg=(ws_bins[1:]+ws_bins[:-1])/2

highlight=(Loads.time.values>=highlight_period[0])&(Loads.time.values<=highlight_period[1])

#%% Main
plt.figure(figsize=(14,5))
for i,v in enumerate(loads_var):
    var=f'{turbine_id}_{v}'
    load=Loads[var]/np.nanmax(Loads[var][highlight])

    load_avg=binned_avg_1d(ws_aligned,load,ws_bins,max_ci_frac)
    valid=~np.isnan(load_avg)#bins meeting the statistical uncertainty requirement

    p10=stats.binned_statistic(ws_aligned,load,statistic=lambda x:np.nanpercentile(x,10),bins=ws_bins)[0]
    p90=stats.binned_statistic(ws_aligned,load,statistic=lambda x:np.nanpercentile(x,90),bins=ws_bins)[0]
    p5= stats.binned_statistic(ws_aligned,load,statistic=lambda x:np.nanpercentile(x,5), bins=ws_bins)[0]
    p95=stats.binned_statistic(ws_aligned,load,statistic=lambda x:np.nanpercentile(x,95),bins=ws_bins)[0]
    p10[~valid]=np.nan; p90[~valid]=np.nan; p5[~valid]=np.nan; p95[~valid]=np.nan

    ax=plt.subplot(1,len(loads_var),i+1)
    # plt.plot(ws_aligned,load,'.k',alpha=0.25,markersize=10,label='Data')
    ws_highlight=ws_aligned[highlight]
    load_highlight=load.values[highlight]
    times_highlight=pd.to_datetime(Loads.time.values[highlight])
    plt.plot(ws_highlight,load_highlight,'.r',markersize=10,label='Selected period')
    for x,y,t in zip(ws_highlight,load_highlight,times_highlight):
        ax.annotate(t.strftime('%H%M'),(x,y),xytext=(0,5),textcoords='offset points',
                    color='r',ha='center',va='bottom',fontsize=9)
    plt.fill_between(ws_avg,p5, p95,color='b',alpha=0.1,label='5$^{th}$-95$^{th}$ percentile')
    plt.fill_between(ws_avg,p10,p90,color='b',alpha=0.2,label='10$^{th}$-90$^{th}$ percentile')
    plt.plot(ws_avg,load_avg,'-b',linewidth=2,label='Bin average')
    plt.xlabel('$U_h$ [m s$^{-1}$]')
    plt.ylabel(labels[v])
    plt.grid()
    if i==0:
        plt.legend()

plt.tight_layout()

os.makedirs(os.path.join(cd,'figures/DEL_stats'),exist_ok=True)
plt.savefig(os.path.join(cd,'figures/DEL_stats',f'{turbine_id}.DEL_stats.png'))

