# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 16:55:29 2025

@author: sletizia
"""
import os
cd=os.path.dirname(__file__)
import matplotlib.pyplot as plt
import numpy as np

import pyart

source=os.path.join(cd,'data','nexrad','KVNX20230805_100440_V06.ar2v')

radar = pyart.io.read_nexrad_archive(source)

gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked("reflectivity")

grid = pyart.map.grid_from_radars(
    (radar,),
    gatefilters=(gatefilter,),
    grid_shape=(1, 241, 241),
    grid_limits=((1000, 1000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
    fields=["reflectivity"],
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolor(grid.x['data'],grid.y['data'],grid.fields["reflectivity"]["data"][0])
plt.show()