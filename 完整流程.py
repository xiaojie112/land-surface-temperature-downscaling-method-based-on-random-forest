# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:34:44 2023

@author: a8362
"""


#先进行夏季建模
import os
import numpy as np
# import earthpy_plot_revised as ep
from osgeo import gdal
import netCDF4 as nc
import pandas as pd

# 读取landsatlst数据以及多种遥感光谱指数并聚合到9km====================================================
variables = {}
basepath =  r'D:\xiaojie\data1\final'
basefiles = os.listdir(basepath) #[ndvi,nbdi,lst,...]
for basefile in basefiles:
    if(basefile == 'era5lst' or basefile == 'output'):continue
    varipath = basepath + '\\' + basefile
    vari9km = np.zeros((4,18,13)) #四幅景
    
    vari5km = np.zeros((4,32,23))
    varifiles = os.listdir(varipath)
    
    vari300m = np.zeros((4,540,390),dtype = np.float64)
    
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
                
        for a in range(32):
            for b in range(23):
                starti = a*166
                startj = b *166
                vari5km[i,a,b] = np.nanmean(ndvi30[starti:starti+166,startj:startj+166].reshape(-1))
                
        for a in range(540):
            for b in range(390):
                starti = a*10
                startj = b *10
                vari300m[i,a,b] = np.nanmean(ndvi30[starti:starti+10,startj:startj+10].reshape(-1))
                
                
                
    variables[basefile] = vari9km
    variables[basefile+'5km'] = vari5km
    variables[basefile+'300m'] = vari300m


landlst = variables['landlst'][0]   #9km数据
mndwi = variables['mndwi'][0]
msavi = variables['msavi'][0]
ndbi = variables['ndbi'][0]
ndmi = variables['ndmi'][0]
ndvi = variables['ndvi'][0]
savi = variables['savi'][0]

print("读取各遥感光谱指数成功!")
# ====================================================
# 读取era5lst数据

era5lst = []

era5path = r'D:\xiaojie\data1\final\era5lst\20130809.nc'
with nc.Dataset(era5path) as file:
    file.set_auto_mask(False)  # 可选
    tempvari = {x: file[x][()] for x in file.variables}
    era5lst = tempvari['skt'][0]
    era5lst -= 273

variables['era5lst'] = era5lst
print("读取era5lst成功!")


# ====================================================
# 构造训练样本
df = pd.DataFrame(data=None,columns=['MNDWI_9KM','MSAVI_9KM','NDBI_9KM','NDMI_9KM','NDVI_9KM','SAVI_9KM','ERA5LST_9KM'])
for row in range(landlst.shape[0]):
    for column in range(landlst.shape[1]):
        if( np.isnan(mndwi[row,column]) 
               or np.isnan(msavi[row,column]) 
               or np.isnan(ndbi[row,column]) 
               or np.isnan(ndmi[row,column]) 
               or np.isnan(ndvi[row,column])
               or np.isnan(savi[row,column])
               or np.isnan(era5lst[row,column]) ):
               continue
        df.loc[len(df.index)] = [mndwi[row,column],msavi[row,column],ndbi[row,column],ndmi[row,column],
                                 ndvi[row,column],savi[row,column],era5lst[row,column]]

df.to_csv(r"D:\xiaojie\data1\sample1.csv",index=False,sep=',')
print("create sample success")




# 训练模型========================================================================
from sklearn.ensemble import RandomForestRegressor # 随机森林回归
# from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error  #回归准确率判断-mse
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np
from math import sqrt

rf = RandomForestRegressor(n_estimators=200, oob_score = True)   

df = pd.read_csv(r"D:\xiaojie\data1\sample1.csv",low_memory = False)
sample = df.to_numpy()

# X,Y = load_boston(return_X_y=True)
X = sample[:,:6]
Y = sample[:,6]
print("read sample success")
x_train, x_validation , y_train, y_validation = train_test_split(X,Y,test_size=0.2)
Test_set = (x_train[1] + x_train[2])/2
rf.fit(x_train,y_train)#训练模型
result_y = rf.predict(x_validation)#测试结果
print("rmse:",mean_squared_error(result_y,y_validation)) # 准确率


# 输入高分辨光谱指数降尺度, 先降尺度到5km
pre =  rf.predict(np.array(Test_set).reshape(1,-1)) #预测



# =============================降尺度到5km====================
# mndwi5km = variables['mndwi5km'][0]
# msavi5km = variables['msavi5km'][0]
# ndbi5km = variables['ndbi5km'][0]
# ndmi5km = variables['ndmi5km'][0]
# ndvi5km = variables['ndvi5km'][0]
# savi5km = variables['savi5km'][0]

# downscalelst = np.zeros((32,23),dtype = np.float64)
# for row in range(32):
#     for column in range(23):
#         if( np.isnan(mndwi5km[row,column]) 
#                or np.isnan(msavi5km[row,column]) 
#                or np.isnan(ndbi5km[row,column]) 
#                or np.isnan(ndmi5km[row,column]) 
#                or np.isnan(ndvi5km[row,column])
#                or np.isnan(savi5km[row,column])):
#             downscalelst[row,column] = np.nan
#             continue
#         prevari = np.array([mndwi5km[row,column],msavi5km[row,column],ndbi5km[row,column],ndmi5km[row,column],
#                    ndvi5km[row,column],savi5km[row,column]]).reshape(1,-1)
#         downscalelst[row,column] = rf.predict(prevari)

# ==========================降尺度到300m
mndwi300m = variables['mndwi300m'][0]
msavi300m = variables['msavi300m'][0]
ndbi300m = variables['ndbi300m'][0]
ndmi300m= variables['ndmi300m'][0]
ndvi300m= variables['ndvi300m'][0]
savi300m= variables['savi300m'][0]

downscalelst = np.zeros((540,390),dtype = np.float64)

df = pd.DataFrame(data=None,columns=['MNDWI_9KM','MSAVI_9KM','NDBI_9KM','NDMI_9KM','NDVI_9KM','SAVI_9KM','ERA5LST_9KM','row', 'colum'])
for row in range(540):
    print("okkkk")
    for column in range(390):
        if( np.isnan(mndwi300m[row,column]) 
               or np.isnan(msavi300m[row,column]) 
               or np.isnan(ndbi300m[row,column]) 
               or np.isnan(ndmi300m[row,column]) 
               or np.isnan(ndvi300m[row,column])
               or np.isnan(savi300m[row,column])):
            downscalelst[row,column] = np.nan
            continue
        df.loc[len(df.index)] = [mndwi300m[row,column],msavi300m[row,column],ndbi300m[row,column],ndmi300m[row,column],
                                 ndvi300m[row,column],savi300m[row,column],np.nan,row, column]
        
        # prevari = np.array([mndwi300m[row,column],msavi300m[row,column],ndbi300m[row,column],ndmi300m[row,column],
        #             ndvi300m[row,column],savi300m[row,column]]).reshape(1,-1)
        # downscalelst[row,column] = rf.predict(prevari)


# ===================================

df_to_arr = np.array(df)
predict = df_to_arr[:,:6]
predict_result = rf.predict(predict)
for i in range(df_to_arr.shape[0]):
    downscalelst[df_to_arr[i,7],df_to_arr[i,8]] = predict_result[i]

# ==============================================================绘图
import os
import numpy as np
import earthpy_plot_revised as ep
import matplotlib.pyplot as plt
from osgeo import gdal

plot_title = "LST300m"


# 忽略nan值求最大和最小值 nanmin nanmax
min_DN = np.nanmin(downscalelst)
max_DN = np.nanmax(downscalelst)
print("min_DN:", min_DN, "\n", "max_DN:", max_DN)

# 栅格数据可视化
ep.plot_bands(downscalelst,
              title=plot_title,
              title_set=[10, 'bold'],
              cmap="seismic",
              # cols=4,
              figsize=(12, 12),
              extent=None,
              cbar=True,
              scale=False,
              vmin= min_DN,
              vmax= max_DN,
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



# landsatlst5km = variables['landlst5km'][0]
# min_DN = np.nanmin(landsatlst5km)
# max_DN = np.nanmax(landsatlst5km)
# ep.plot_bands(landsatlst5km,
#               title=plot_title,
#               title_set=[10, 'bold'],
#               cmap="seismic",
#               # cols=4,
#               figsize=(12, 12),
#               extent=None,
#               cbar=True,
#               scale=False,
#               vmin= min_DN,
#               vmax= max_DN,
#               ax=None,
#               alpha=1,
#               norm=None,
#               save_or_not=False,
#               # save_path=output_file_path_im,
#               dpi_out=300,
#               bbox_inches_out="tight",
#               pad_inches_out=10,
#               text_or_not=True,
#               text_set=[1.2, 0.95, "q(°C)", 3, 'bold'],
#               # colorbar_label_set=True,
#               label_size=50,
#               cbar_len=50,
#               )

























