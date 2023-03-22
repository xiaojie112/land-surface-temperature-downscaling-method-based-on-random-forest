# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:50:10 2023

@author: a8362
"""


# 参考链接https://cloud.tencent.com/developer/article/1677026
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os

# tif01 =r"D:\xiaojie\data1\final\landlst\20130809_lst.tif"
# tifdata01 = gdal.Open(tif01)
# rows = tifdata01.RasterXSize
# columns = tifdata01.RasterYSize
# bands = tifdata01.RasterCount
# rows,columns,bands


# #获取地理信息
# img_geotrans = tifdata01.GetGeoTransform()
# #获取投影信息
# img_proj = tifdata01.GetProjection()



variables = {}

# 读取NDVI(5136,3846)->(18,13)
basepath =  r'D:\xiaojie\data1\final'
basefiles = os.listdir(basepath) #[ndvi,nbdi,lst,...]
for basefile in basefiles:
    if(basefile == 'era5lst' or basefile == 'output'):continue
    varipath = basepath + '\\' + basefile
    vari9km = np.zeros((4,18,13)) #四幅景,春夏秋冬
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




#获取对应的数据
# tif_data01 = tifdata01.ReadAsArray(0,0,rows,columns)
tif_data01 = variables['landlst'][0]
# tif_data01 = tif_data01.astype('float64')
# tif_data01[tif_data01 == 32767] = np.nan
# tif_data01[tif_data01 == 0] = np.nan
# tif_data01 = tif_data01/10-273

# ========================================================
# 读取高分辨率数据

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
# lst[lst > 40] = np.nan
tif_data02 = lst

# ==========================================================



min_list = []
max_list = []


min_list2 = []
max_list2 = []


min_list.append(np.nanmin(tif_data01))
min_list.append(np.nanmin(tif_data02))
# min_list.append(np.nanmin(tif_data03))
data_min = np.min(min_list)

max_list.append(np.nanmax(tif_data01))
max_list.append(np.nanmax(tif_data02))
# max_list.append(np.nanmax(tif_data03))

data_max = np.max(max_list)



# fig,ax = plt.subplots(1, 3,figsize = (17,10),sharey=True)
fig,ax = plt.subplots(1, 3,figsize = (17,10))


vmin = data_min
vmax = data_max
#Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，
#颜色表/颜色柱的起始和终止分别取值vmin和vmax
# norm = Normalize(vmin = vmin,vmax = vmax)
# extent = (112,114,22,24)

# im1 = ax[0].imshow(tif_data01,extent = extent, cmap = 'jet')

im1 = ax[0].imshow(tif_data01, cmap = 'jet', vmin = 30, vmax = 35)

# ax[0].set_axis_off()
im2 = ax[1].imshow(tif_data02,cmap = 'jet', vmin = 26, vmax = 39)
# ax[1].set_axis_off()
# im3 = ax[2].imshow(tif_data03,extent = extent,norm = norm,cmap = 'jet')
# ax[2].set_axis_off()
ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
        ha='center', va='center',fontsize = 10,color='black')

fig.subplots_adjust(right=1)

#前面三个子图的总宽度为全部宽度的 0.9；剩下的0.1用来放置colorbar
fig.subplots_adjust(right=0.9)
position = fig.add_axes([0.9, 0.22, 0.015, .55 ])#位置[左,下,右,上]
cb = fig.colorbar(im1, cax=position)

#设置colorbar标签字体等
colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
cb.ax.set_title('Values',fontdict=colorbarfontdict,pad=8)
cb.ax.set_ylabel('EvapotTranspiration(ET)',fontdict=colorbarfontdict)
cb.ax.tick_params(labelsize=11,direction='in')
#cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
fig.suptitle('One Colorbar for Multiple Map Plot ',size=22,family='Times New Roman',
             x=.55,y=.9)
# plt.savefig(r'F:\DataCharm\Python-matplotlib 空间数据可视化\map_colorbar.png',dpi = 600,
            # bbox_inches='tight',width = 12,height=4)
plt.show()