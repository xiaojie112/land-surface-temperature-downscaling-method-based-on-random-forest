# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:12:30 2023

@author: a8362
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import earthpy_plot_revised as ep
from osgeo import gdal



# 文件目录
root_path = r"D:\xiaojie\data1\final"
out_path = r"D:\xiaojie\data1\final\output"

# 输入路径
for filepath, dirnames, filenames in os.walk(root_path):
    for filename in filenames:
        if filename.endswith(".tif"):
            tif_file_path = os.path.join(filepath, filename)
            print(tif_file_path)

            # 输出路径
            output_file_path_im = os.path.join(out_path,
                                               f"{filename[0:4]}_T.jpg")

            plot_title = f"{filename[0:4]}_LST"
            # 打开图像，并读取为数组
            t_raster = gdal.Open(tif_file_path)
            raster_arr = t_raster.ReadAsArray()

            # 将图像的背景值设置为nan
            raster_arr[raster_arr < -16] = 'nan'

            # 忽略nan值求最大和最小值 nanmin nanmax
            min_DN = np.nanmin(raster_arr)
            max_DN = np.nanmax(raster_arr)
            print("min_DN:", min_DN, "\n", "max_DN:", max_DN)

            # 栅格数据可视化
            ep.plot_bands(raster_arr,
                          title=plot_title,
                          title_set=[25, 'bold'],
                          # cmap="seismic",
                          cols=3,
                          figsize=(12, 12),
                          extent=None,
                          cbar=True,
                          scale=False,
                          vmin= min_DN,
                          vmax= max_DN,
                          ax=None,
                          alpha=1,
                          norm=None,
                          save_or_not=True,
                          save_path=output_file_path_im,
                          dpi_out=300,
                          bbox_inches_out="tight",
                          pad_inches_out=0.1,
                          text_or_not=True,
                          text_set=[0.75, 0.95, "T(°C)", 20, 'bold'],
                          colorbar_label_set=True,
                          label_size=20,
                          cbar_len=2,
                          )

# 蓝到红的渐变 三种 seismic bwr coolwarm
# rainbow jet RdBu_r RdYIBu_r
# x_pos, y_pos 距离左上角的距离; text_content
# text_set=[0.98,0.98,"°C",15]
# x, y, text_content, fontsize=15, fontweight='normal'
