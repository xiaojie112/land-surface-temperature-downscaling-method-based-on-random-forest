# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:19:43 2023

@author: a8362
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 11:11:24 2023

@author: a8362
"""

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
# In[]
# 读取landsatlst数据以及多种遥感光谱指数并聚合到9km====================================================
variables = {}
basepath =  r'D:\xiaojie\data1\spring'
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
            
            
    for a in range(337):
        for b in range(243):
            starti = a*16
            startj = b *16
            vari500m[a,b] = np.nanmean(ndvi30[starti:starti+16,startj:startj+16].reshape(-1))
                  
                  
    variables[basefile+'9km'] = vari9km
    variables[basefile+'5km'] = vari5km
    variables[basefile+'300m'] = vari300m
    variables[basefile+'2km'] = vari2km
    variables[basefile+'500m'] = vari500m



landlst = variables['landlst'+'9km']   #9km数据,由30m分辨率的landlst聚合得到
mndwi = variables['mndwi'+'9km']
msavi = variables['msavi'+'9km']
ndbi = variables['ndbi'+'9km']
ndmi = variables['ndmi'+'9km']
ndvi = variables['ndvi'+'9km']
savi = variables['savi'+'9km']

print("读取各遥感光谱指数成功!")




# ====================================================
# In[]  读取era5lst数据

era5lst = []

era5path = r'D:\xiaojie\data1\spring\era5lst\20180401.nc'
with nc.Dataset(era5path) as file:
    file.set_auto_mask(False)  # 可选
    tempvari = {x: file[x][()] for x in file.variables}
    era5lst = tempvari['skt'][0]
    era5lst -= 273

variables['era5lst'] = era5lst
print("读取era5lst成功!")


# ====================================================
# In[]
# 构造训练样本   9km变量训练模型
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

df.to_csv(r"D:\xiaojie\data1\spring_sample9km.csv",index=False,sep=',')
print("create sample success")



# In[]
# 训练模型  9km变量训练模型========================================================================

from sklearn.ensemble import RandomForestRegressor # 随机森林回归
# from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error  #回归准确率判断-mse
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np
from math import sqrt

def rmse(predictions, targets):
    return sqrt(np.nanmean(((predictions - targets) ** 2)))


def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

# rf = RandomForestRegressor(n_estimators=200, oob_score = True)   
rf = RandomForestRegressor(n_estimators=77,random_state=0,max_depth=5,max_features=3)   


df = pd.read_csv(r"D:\xiaojie\data1\spring_sample9km.csv",low_memory = False)
sample = df.to_numpy()

# X,Y = load_boston(return_X_y=True)
X = sample[:,:6]
Y = sample[:,6]
print("read sample success")
x_train, x_validation , y_train, y_validation = train_test_split(X,Y,test_size=0.2)
# Test_set = (x_train[1] + x_train[2])/2
rf.fit(x_train,y_train)#训练模型
result_y = rf.predict(x_validation)#测试结果
print("rmse:",mean_squared_error(result_y,y_validation)) # 准确率
print("rmse2:",rmse(result_y, y_validation))



# 输入高分辨光谱指数降尺度, 先降尺度到5km
# pre =  rf.predict(np.array(Test_set).reshape(1,-1)) #预测


# In[] 
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


# In[]降尺度到300m
mndwi300m = variables['mndwi300m']
msavi300m = variables['msavi300m']
ndbi300m = variables['ndbi300m']
ndmi300m= variables['ndmi300m']
ndvi300m= variables['ndvi300m']
savi300m= variables['savi300m']
# 这里的downscalelst对应降尺度后的era5lst
downscalelst300m = np.zeros((540,390),dtype = np.float64)

df = pd.DataFrame(data=None,columns=['MNDWI_9KM','MSAVI_9KM','NDBI_9KM','NDMI_9KM','NDVI_9KM','SAVI_9KM','ERA5LST_9KM','row', 'colum'])
for row in range(540):
    print(row)
    for column in range(390):
        if( np.isnan(mndwi300m[row,column]) 
               or np.isnan(msavi300m[row,column]) 
               or np.isnan(ndbi300m[row,column]) 
               or np.isnan(ndmi300m[row,column]) 
               or np.isnan(ndvi300m[row,column])
               or np.isnan(savi300m[row,column])):
            downscalelst300m[row,column] = np.nan
            continue
        df.loc[len(df.index)] = [mndwi300m[row,column],msavi300m[row,column],ndbi300m[row,column],ndmi300m[row,column],
                                 ndvi300m[row,column],savi300m[row,column],np.nan,row, column]
        
        # prevari = np.array([mndwi300m[row,column],msavi300m[row,column],ndbi300m[row,column],ndmi300m[row,column],
        #             ndvi300m[row,column],savi300m[row,column]]).reshape(1,-1)
        # downscalelst[row,column] = rf.predict(prevari)


# ===================================
# In[]  填充300m

df_to_arr = np.array(df)
predict = df_to_arr[:,:6]
predict_result = rf.predict(predict)
for i in range(df_to_arr.shape[0]):
    downscalelst300m[int(df_to_arr[i,7]),int(df_to_arr[i,8])] = predict_result[i]


# In[] 降尺度到500m

mndwi500m = variables['mndwi500m']
msavi500m = variables['msavi500m']
ndbi500m = variables['ndbi500m']
ndmi500m= variables['ndmi500m']
ndvi500m= variables['ndvi500m']
savi500m= variables['savi500m']

downscalelst500m = np.zeros((337,243),dtype = np.float64)

df = pd.DataFrame(data=None,columns=['MNDWI_9KM','MSAVI_9KM','NDBI_9KM','NDMI_9KM','NDVI_9KM','SAVI_9KM','ERA5LST_9KM','row', 'colum'])
for row in range(337):
    print(row)
    for column in range(243):
        if( np.isnan(mndwi500m[row,column]) 
               or np.isnan(msavi500m[row,column]) 
               or np.isnan(ndbi500m[row,column]) 
               or np.isnan(ndmi500m[row,column]) 
               or np.isnan(ndvi500m[row,column])
               or np.isnan(savi500m[row,column])):
            downscalelst500m[row,column] = np.nan
            continue
        df.loc[len(df.index)] = [mndwi500m[row,column],msavi500m[row,column],ndbi500m[row,column],ndmi500m[row,column],
                                 ndvi500m[row,column],savi500m[row,column],np.nan,row, column]
        
        # prevari = np.array([mndwi300m[row,column],msavi300m[row,column],ndbi300m[row,column],ndmi300m[row,column],
        #             ndvi300m[row,column],savi300m[row,column]]).reshape(1,-1)
        # downscalelst[row,column] = rf.predict(prevari)


# In[]  填充500m

df_to_arr = np.array(df)
predict = df_to_arr[:,:6]
predict_result = rf.predict(predict)
for i in range(df_to_arr.shape[0]):
    downscalelst500m[int(df_to_arr[i,7]),int(df_to_arr[i,8])] = predict_result[i]


# In[] 降尺度到2km

mndwi2km = variables['mndwi2km']
msavi2km = variables['msavi2km']
ndbi2km = variables['ndbi2km']
ndmi2km = variables['ndmi2km']
ndvi2km = variables['ndvi2km']
savi2km = variables['savi2km']

downscalelst2km = np.zeros((81,59),dtype = np.float64)

df = pd.DataFrame(data=None,columns=['MNDWI_9KM','MSAVI_9KM','NDBI_9KM','NDMI_9KM','NDVI_9KM','SAVI_9KM','ERA5LST_9KM','row', 'colum'])
for row in range(81):
    print(row)
    for column in range(59):
        if( np.isnan(mndwi2km[row,column]) 
               or np.isnan(msavi2km[row,column]) 
               or np.isnan(ndbi2km[row,column]) 
               or np.isnan(ndmi2km[row,column]) 
               or np.isnan(ndvi2km[row,column])
               or np.isnan(savi2km[row,column])):
            downscalelst2km[row,column] = np.nan
            continue
        df.loc[len(df.index)] = [mndwi2km[row,column],msavi2km[row,column],ndbi2km[row,column],ndmi2km[row,column],
                                 ndvi2km[row,column],savi2km[row,column],np.nan,row, column]


# In[] 填充2km
df_to_arr = np.array(df)
predict = df_to_arr[:,:6]
predict_result = rf.predict(predict)
for i in range(df_to_arr.shape[0]):
    downscalelst2km[int(df_to_arr[i,7]),int(df_to_arr[i,8])] = predict_result[i]
    



# In[] landsat30m聚合到landsat500m
# dataset = gdal.Open(r'D:\xiaojie\data1\final\landlst\20130809_lst.tif')
# band = dataset.GetRasterBand(1)
# landsatlst30m = np.zeros((5400,3900))
# data = band.ReadAsArray() #30m分辨率,5136*3846
# landsatlst30m[:data.shape[0],:data.shape[1]] = data
# landsatlst30m[landsatlst30m == 32767] = np.nan
# landsatlst30m[landsatlst30m == 0] = np.nan
# landsatlst30m = landsatlst30m/10-273

landsatlst500m = variables['landlst500m']
landsatlst2km = variables['landlst2km']
landsat300m = variables['landlst300m']
# downscalelst300m = np.load(r'D:\xiaojie\data1\final\output\code\downscalelst300m.npy')



# In[] 计算rmse和r       300m
def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

from scipy.stats import pearsonr

# 已有predict_result
origin_result = np.zeros(predict_result.shape[0])
for i in range(predict_result.shape[0]):
    origin_result[i] = landsat300m[int(df_to_arr[i,7]),int(df_to_arr[i,8])]

indexNotNan = ~np.array([np.isnan(origin_result)])[0]
origin_result = origin_result[indexNotNan]

predict_result = predict_result[indexNotNan]

print("rmse:",rmse(predict_result, origin_result))
# print("r:",cal_pccs(predict_result,, origin_result, predict_result.shape[0]))
ret = cal_pccs(predict_result,origin_result, predict_result.shape[0])



print(cal_pccs(predict_result,origin_result, predict_result.shape[0]))
print(pearsonr(predict_result, origin_result))







# In[] 计算rmse和r       2km
def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

from scipy.stats import pearsonr

# 已有predict_result
origin_result = np.zeros(predict_result.shape[0])
for i in range(predict_result.shape[0]):
    origin_result[i] = landsatlst500m[int(df_to_arr[i,7]),int(df_to_arr[i,8])]

indexNotNan = ~np.array([np.isnan(origin_result)])[0]
origin_result = origin_result[indexNotNan]

predict_result = predict_result[indexNotNan]

print("rmse:",rmse(predict_result, origin_result))
# print("r:",cal_pccs(predict_result,, origin_result, predict_result.shape[0]))
ret = cal_pccs(predict_result,origin_result, predict_result.shape[0])



print(cal_pccs(predict_result,origin_result, predict_result.shape[0]))
print(pearsonr(predict_result, origin_result))

# In[] 绘图

# from osgeo import gdal
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# #获取对应的数据

# tif_data01 = landsat300m[20:landsat300m.shape[0]-35,10:landsat300m.shape[1]-15]


# tif_data02 = downscalelst300m[20:downscalelst300m.shape[0]-35,10:downscalelst300m.shape[1]-15]

# tif_data03 =  landsatlst500m[20:landsatlst500m.shape[0]-35,10:landsatlst500m.shape[1]-15]

# tif_data04 = downscalelst500m[20:downscalelst500m.shape[0]-35,10:downscalelst500m.shape[1]-20]

# tif_data05 = landsatlst2km[5:landsatlst2km.shape[0]-15,5:landsatlst2km.shape[1]-8]

# tif_data06 = downscalelst2km[5:downscalelst2km.shape[0]-15,5:downscalelst2km.shape[1]-7]
# # ==========================================================



# min_list = []
# max_list = []


# min_list2 = []
# max_list2 = []


# min_list.append(np.nanmin(tif_data01))
# min_list.append(np.nanmin(tif_data02))
# # min_list.append(np.nanmin(tif_data03))
# data_min = np.min(min_list)

# max_list.append(np.nanmax(tif_data01))
# max_list.append(np.nanmax(tif_data02))
# # max_list.append(np.nanmax(tif_data03))

# data_max = np.max(max_list)



# # fig,ax = plt.subplots(1, 3,figsize = (17,10),sharey=True)
# fig,ax = plt.subplots(2, 3,figsize = (10,7))


# vmin = data_min
# vmax = data_max
# #Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，
# #颜色表/颜色柱的起始和终止分别取值vmin和vmax
# # norm = Normalize(vmin = vmin,vmax = vmax)
# # extent = (112,114,22,24)

# # im1 = ax[0].imshow(tif_data01,extent = extent, cmap = 'jet')


# ax[0][0].set_xticks([])
# ax[0][0].set_yticks([])

# ax[0][1].set_xticks([])
# ax[0][1].set_yticks([])

# ax[0][2].set_xticks([])
# ax[0][2].set_yticks([])


# ax[1][0].set_xticks([])
# ax[1][0].set_yticks([])

# ax[1][1].set_xticks([])
# ax[1][1].set_yticks([])

# ax[1][2].set_xticks([])
# ax[1][2].set_yticks([])

# im1 = ax[0][0].imshow(tif_data01, cmap = 'jet', vmin = np.nanmin(5), vmax = np.nanmax(40))

# im2 = ax[0][1].imshow(tif_data02,cmap = 'jet', vmin = np.nanmin(tif_data02), vmax = np.nanmax(tif_data02))

# im3 = ax[0][2].imshow(tif_data03,cmap = 'jet', vmin = 5, vmax = 40)

# im4 = ax[1][0].imshow(tif_data04,cmap = 'jet', vmin = np.nanmin(tif_data04), vmax = np.nanmax(tif_data04))

# im5 = ax[1][1].imshow(tif_data05,cmap = 'jet', vmin = 5, vmax = 40)

# im6 = ax[1][2].imshow(tif_data06,cmap = 'jet', vmin = np.nanmin(tif_data06), vmax = np.nanmax(tif_data06))



# # ax[1].set_axis_off()
# # im3 = ax[2].imshow(tif_data03,extent = extent,norm = norm,cmap = 'jet')
# # ax[2].set_axis_off()

# # ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
# #         ha='center', va='center',fontsize = 10,color='black')

# # fig.subplots_adjust(right=1)==========

# #前面三个子图的总宽度为全部宽度的 0.9；剩下的0.1用来放置colorbar
# # fig.subplots_adjust(right=0.9)
# # position = fig.add_axes([0.9, 0.22, 0.015, .55 ])#位置[左,下,右,上]
# # cb = fig.colorbar(im1, ax= ax[1][0], cax=position, extend)
# cb = fig.colorbar(im1, ax= ax[0][0]) # extend的作用
# cb.ax.set_title('/℃')
# cb = fig.colorbar(im2,ax=ax[0][1])
# cb.ax.set_title('/℃')

# cb = fig.colorbar(im3,ax=ax[0][2])
# cb.ax.set_title('/℃')

# cb =fig.colorbar(im4,ax=ax[1][0])
# cb.ax.set_title('/℃')

# cb =fig.colorbar(im5,ax=ax[1][1])
# cb.ax.set_title('/℃')

# cb =fig.colorbar(im6,ax=ax[1][2])
# cb.ax.set_title('/℃')


# #设置colorbar标签字体等
# # colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
# # cb.ax.set_title('Values',fontdict=colorbarfontdict,pad=8)
# # cb.ax.set_ylabel('EvapotTranspiration(ET)',fontdict=colorbarfontdict)
# # cb.ax.tick_params(labelsize=11,direction='in')

# #cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
# # fig.suptitle('One Colorbar for Multiple Map Plot ',size=22,family='Times New Roman',
#              # x=.55,y=.9)
# # plt.savefig(r'F:\DataCharm\Python-matplotlib 空间数据可视化\map_colorbar.png',dpi = 600,
#             # bbox_inches='tight',width = 12,height=4)
# plt.show()

# In[]

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os

#获取对应的数据

tif_data01 = landsat300m[20:landsat300m.shape[0]-35,10:landsat300m.shape[1]-15]


tif_data02 = downscalelst300m[20:downscalelst300m.shape[0]-35,10:downscalelst300m.shape[1]-15]

tif_data03 =  landsatlst500m[20:landsatlst500m.shape[0]-35,10:landsatlst500m.shape[1]-15]

tif_data04 = downscalelst500m[20:downscalelst500m.shape[0]-35,10:downscalelst500m.shape[1]-20]

tif_data05 = landsatlst2km[5:landsatlst2km.shape[0]-15,5:landsatlst2km.shape[1]-8]

tif_data06 = downscalelst2km[5:downscalelst2km.shape[0]-15,5:downscalelst2km.shape[1]-7]
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
fig,ax = plt.subplots(1, 3,figsize = (10,3))


vmin = data_min
vmax = data_max
#Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，
#颜色表/颜色柱的起始和终止分别取值vmin和vmax
# norm = Normalize(vmin = vmin,vmax = vmax)
# extent = (112,114,22,24)

# im1 = ax[0].imshow(tif_data01,extent = extent, cmap = 'jet')


ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].set_xticks([])
ax[2].set_yticks([])





im4 = ax[0].imshow(tif_data04,cmap = 'jet', vmin = np.nanmin(tif_data04), vmax = np.nanmax(tif_data04))

im5 = ax[1].imshow(tif_data05,cmap = 'jet', vmin = 5, vmax = 40)

im6 = ax[2].imshow(tif_data06,cmap = 'jet', vmin = np.nanmin(tif_data06), vmax = np.nanmax(tif_data06))



# ax[1].set_axis_off()
# im3 = ax[2].imshow(tif_data03,extent = extent,norm = norm,cmap = 'jet')
# ax[2].set_axis_off()

# ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
#         ha='center', va='center',fontsize = 10,color='black')

# fig.subplots_adjust(right=1)==========

#前面三个子图的总宽度为全部宽度的 0.9；剩下的0.1用来放置colorbar
# fig.subplots_adjust(right=0.9)
# position = fig.add_axes([0.9, 0.22, 0.015, .55 ])#位置[左,下,右,上]
# cb = fig.colorbar(im1, ax= ax[1][0], cax=position, extend)


cb =fig.colorbar(im4,ax=ax[0])
cb.ax.set_title('/℃')

cb =fig.colorbar(im5,ax=ax[1])
cb.ax.set_title('/℃')

cb =fig.colorbar(im6,ax=ax[2])
cb.ax.set_title('/℃')


#设置colorbar标签字体等
# colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
# cb.ax.set_title('Values',fontdict=colorbarfontdict,pad=8)
# cb.ax.set_ylabel('EvapotTranspiration(ET)',fontdict=colorbarfontdict)
# cb.ax.tick_params(labelsize=11,direction='in')

#cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
# fig.suptitle('One Colorbar for Multiple Map Plot ',size=22,family='Times New Roman',
              # x=.55,y=.9)
# plt.savefig(r'F:\DataCharm\Python-matplotlib 空间数据可视化\map_colorbar.png',dpi = 600,
            # bbox_inches='tight',width = 12,height=4)
plt.show()












