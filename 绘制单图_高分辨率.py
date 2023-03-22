# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:08:38 2023

@author: a8362
"""

import os
import numpy as np
import earthpy_plot_revised as ep
from osgeo import gdal



# 文件目录
root_path = r"D:\xiaojie\data1\final"
out_path = r"D:\xiaojie\data1\final\output"


tif_file_path = r"D:\xiaojie\data1\final\landlst\20130809_lst.tif"
print(tif_file_path)

# 输出路径
# output_file_path_im = r'D:\xiaojie\data1\final\output\'

plot_title = "LST"
# 打开图像，并读取为数组
t_raster = gdal.Open(tif_file_path)
data = t_raster.ReadAsArray()

# 将图像的背景值设置为nan
# lst = np.zeros((5136,3846))
lst = np.zeros((data.shape[0]-200,data.shape[1]-200),dtype=np.float64)


lst[:,:] = data[100:data.shape[0]-100,100:data.shape[1]-100]

lst[lst == 32767] = np.nan
lst[lst == 0] = np.nan
lst = lst/10-273





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
              figsize=(6, 6),
              extent=None,
              cbar=True,
              scale=False,
              vmin= 15,
              vmax= 45,
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