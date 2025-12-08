# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 18:51:18 2025

@author: sletizia
"""
import os
from PIL import Image

#%% Inputs
source="figures/KNVX-20230805-0900-1200-ANI.gif"

#%% Initialization

# Open GIF
gif = Image.open(source)

frame_number = 0
try:
    while True:
        # Save current frame as PNG
        frame_path = os.path.join("figures", f"frame_{frame_number:03d}.png")
        gif.save(frame_path, format="PNG")
        
        # Move to next frame
        frame_number += 1
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

