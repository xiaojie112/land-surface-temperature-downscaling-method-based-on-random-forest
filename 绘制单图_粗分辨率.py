# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:20:49 2023

@author: a8362
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:08:38 2023

@author: a8362
"""

import os
import numpy as np
import earthpy_plot_revised as ep
import matplotlib.pyplot as plt
from osgeo import gdal




variables = {}

# 读取NDVI(5136,3846)->(18,13)
basepath =  r'D:\xiaojie\data1\final'
basefiles = os.listdir(basepath) #[ndvi,nbdi,lst,...]
for basefile in basefiles:
    if(basefile == 'era5lst' or basefile == 'output'):continue
    varipath = basepath + '\\' + basefile
    vari9km = np.zeros((4,18,13)) #四幅景
    varifiles = os.listdir(varipath)
    for i in range(len(varifiles)):
        ndvipath = varipath + '\\' + varifiles[i]
        dataset = gdal.Open(ndvipath)
        print(dataset.GetProjection())
        print("============================================================")
        band = dataset.GetRasterBand(1)
        ndvi30 = np.zeros((5400,3900))
        data = band.ReadAsArray() #30m分辨率,5136*3846
        ndvi30[:data.shape[0],:data.shape[1]] = data
        if(basefile == 'landlst'):
            ndvi30[ndvi30 == 32767] = np.nan
            ndvi30[ndvi30 == 0] = np.nan
            ndvi30 = ndvi30/10-273
        else:
            ndvi30[ndvi30 == 0] = np.nan
        for a in range(18):
            for b in range(13):
                starti = a*300
                startj = b *300
                vari9km[i,a,b] = np.nanmean(ndvi30[starti:starti+300,startj:startj+300].reshape(-1))
                
    variables[basefile] = vari9km

lst = variables['landlst'][0]


plot_title = "LST"


# 忽略nan值求最大和最小值 nanmin nanmax
min_DN = np.nanmin(lst)
max_DN = np.nanmax(lst)
print("min_DN:", min_DN, "\n", "max_DN:", max_DN)

# 栅格数据可视化
ep.plot_bands(lst,
              title=plot_title,
              title_set=[10, 'bold'],
              cmap="seismic",
              # cols=4,
              figsize=(12, 12),
              extent=None,
              cbar=True,
              scale=False,
              vmin= 20,
              vmax= 40,
              ax=None,
              alpha=1,
              norm=None,
              save_or_not=False,
              # save_path=output_file_path_im,
              dpi_out=300,
              bbox_inches_out="tight",
              pad_inches_out=10,
              text_or_not=True,
              text_set=[1.2, 0.95, "q(°C)", 3, 'bold'],
              # colorbar_label_set=True,
              label_size=50,
              cbar_len=50,
              )