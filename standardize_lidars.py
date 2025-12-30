# -*- coding: utf-8 -*-
'''
Processor of lidars through LIDARGO

Inputs (both hard-coded and available as command line inputs in this order):
    sdate [%Y-%m-%d]: start date in UTC
    edate [%Y-%m-%d]: end date in UTC
    delete [bool]: whether to delete raw data
    path_config: path to general config file
    mode [str]: serial or parallel
'''
import os
cd=os.path.dirname(__file__)
import sys
import traceback
import warnings
import lidargo as lg
from datetime import datetime
import yaml
from multiprocessing import Pool
import logging
import re
import glob
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2023-08-05' #start date
    edate='2023-08-06' #end date
    delete=False #delete input files?
    replace=True #replace existing files?
    path_config=os.path.join(cd,'configs/config.yaml') #config path
    mode='serial'#serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2] 
    delete=sys.argv[3]=="True"
    replace=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]#
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)

#%% Functions
def standardize_file(file,save_path_stand,config,logfile_main,sdate,edate,delete,replace):
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
            lproc = lg.Standardize(file, config=config['path_config_stand'], verbose=True,logfile=logfile)
            lproc.process_scan(replace=replace, save_file=True, save_path=save_path_stand)
            if delete:
                os.remove(file)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')

#%% Main
for s in config['channels']:
        
    #standardize all files within date range
    channel=config['channels'][s]
    files=sorted(glob.glob(os.path.join(config['path_data'],channel,config['wildcard_stand'][s])))
    if mode=='serial':
        for f in files:
              standardize_file(f,None,config,logfile_main,sdate,edate,delete,replace)
    elif mode=='parallel':
        args = [(files[i],None, config,logfile_main,sdate,edate,delete,replace) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(standardize_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
          
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
