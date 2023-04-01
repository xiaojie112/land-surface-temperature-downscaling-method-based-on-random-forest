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
    varifiles = os.listdir(varipath)



    vari9km = np.zeros((18,13)) #四幅景
    vari5km = np.zeros((32,23))
    vari2km = np.zeros((81,59),dtype=np.float64)
    vari500m = np.zeros((337,243),dtype=np.float64)
    vari300m = np.zeros((540,390),dtype = np.float64)
    
    ndvipath = varipath + '\\' + varifiles[0]
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
            vari9km[a,b] = np.nanmean(ndvi30[starti:starti+300,startj:startj+300].reshape(-1))
            
    for a in range(32):
        for b in range(23):
            starti = a*166
            startj = b *166
            vari5km[a,b] = np.nanmean(ndvi30[starti:starti+166,startj:startj+166].reshape(-1))
            
    for a in range(540):
        for b in range(390):
            starti = a*10
            startj = b *10
            vari300m[a,b] = np.nanmean(ndvi30[starti:starti+10,startj:startj+10].reshape(-1))
            
    for a in range(81):
        for b in range(59):
            starti = a*66
            startj = b *66
            vari2km[a,b] = np.nanmean(ndvi30[starti:starti+66,startj:startj+66].reshape(-1))
            
            
    for a in range(81):
        for b in range(59):
            starti = a*16
            startj = b *16
            vari500m[a,b] = np.nanmean(ndvi30[starti:starti+16,startj:startj+16].reshape(-1))
                  
                  
    variables[basefile+'9km'] = vari9km
    variables[basefile+'5km'] = vari5km
    variables[basefile+'300m'] = vari300m
    variables[basefile+'2km'] = vari2km
    variables[basefile+'500m'] = vari2km



landlst = variables['landlst'+'9km']   #9km数据
mndwi = variables['mndwi'+'9km']
msavi = variables['msavi'+'9km']
ndbi = variables['ndbi'+'9km']
ndmi = variables['ndmi'+'9km']
ndvi = variables['ndvi'+'9km']
savi = variables['savi'+'9km']

print("读取各遥感光谱指数成功!")


# In[]
# 训练模型  9km变量训练模型========================================================================

from sklearn.ensemble import RandomForestRegressor # 随机森林回归
# from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error  #回归准确率判断-mse
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np
from math import sqrt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# rf = RandomForestRegressor(n_estimators=200, oob_score = True)   
rf = RandomForestRegressor(n_estimators=100,random_state=0)   
df = pd.read_csv(r"D:\xiaojie\data1\sample9km.csv",low_memory = False)
sample = df.to_numpy()

# X,Y = load_boston(return_X_y=True)
X = sample[:,:6]
Y = sample[:,6]
print("read sample success")
x_train, x_validation , y_train, y_validation = train_test_split(X,Y,test_size=0.2)

score_pre = cross_val_score(rf, X, Y, cv=3,scoring = "neg_mean_squared_error").mean()
score_pre

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,200,10):
    rf = RandomForestRegressor(n_estimators=i+1,random_state=0) 
    score = cross_val_score(rf, X, Y, cv=3, scoring = "neg_mean_squared_error").mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))

# 绘制学习曲线
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()


# In[]
# 在41附近缩小n_estimators的范围为30-49
score_lt = []
for i in range(70,90):
    rf = RandomForestRegressor(n_estimators=i+1,random_state=0) 
    score = cross_val_score(rf, X, Y, cv=10, scoring = "neg_mean_squared_error").mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)+70))

# 绘制学习曲线
x = np.arange(70,90)
plt.subplot(111)
plt.plot(x, score_lt,'o-')
plt.show()


# In[]
# 建立n_estimators为45的随机森林
rf = RandomForestRegressor(n_estimators=77,random_state=0) 

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rf, param_grid, cv=10,scoring = "neg_mean_squared_error")
GS.fit(X, Y)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

# In[]

# 用网格搜索调整max_features
param_grid = {'max_features':np.arange(2,7)}

rf = RandomForestRegressor(n_estimators=77,random_state=0,max_depth=5) 

GS = GridSearchCV(rf, param_grid, cv=10,scoring = "neg_mean_squared_error")
GS.fit(X, Y)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)  