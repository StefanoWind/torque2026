# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 16:55:29 2025

@author: sletizia
"""
def change_color(turbine_file,color):
    from PIL import Image
    img = Image.open(turbine_file).convert("RGBA")

    pixels = img.load()
    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            # Detect black (or near-black)
            if a > 0 and r < 50 and g < 50 and b < 50:
                pixels[x, y] = (*color, a)
    return img            
    
def draw_turbine(x,y,D,wd,turbine_file,color=(255, 0, 0)):
    from matplotlib import transforms
    from matplotlib import pyplot as plt
    img=change_color(turbine_file,color)
    
    # img = mpimg.imread("tmp.png")
    ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    tr = transforms.Affine2D().scale(D/1800,D/1800).translate(-20*D/700,-350*D/700).rotate_deg(90-wd).translate(x,y)
    ax.imshow(img, transform=tr + ax.transData,zorder=10)
    plt.xlim(xlim)
    plt.ylim(ylim)
    

draw_turbine(0,0, 1, 0,'./figures/turbine.png',color=[100,100,2])