#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:32:49 2023

@author: damon
"""

# -*- encoding: utf-8 -*-

import xarray as xr
import glob2
import os
from osgeo import gdal
from osgeo import osr
import numpy as np
import math

import os
os.environ['PROJ_LIB'] = r'C:\Users\a8362\anaconda3\Lib\site-packages\osgeo\data\proj'


def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def point_transform(source_ref,target_ref,x,y):
    #创建目标空间参考
    spatialref_target=osr.SpatialReference()
    spatialref_target.ImportFromEPSG(target_ref) 
    #创建原始空间参考
    spatialref_source=osr.SpatialReference()
    spatialref_source.ImportFromEPSG(source_ref)  #WGS84
    #构建坐标转换对象，用以转换不同空间参考下的坐标
    trans=osr.CoordinateTransformation(spatialref_source,spatialref_target)
    coordinate_after_trans=trans.TransformPoint(x,y)
    return coordinate_after_trans
	 #以下为转换多个点（要使用list）
    #coordinate_trans_points=trans.TransformPoints([(117,40),(120,36)])
    #print(coordinate_trans_points)






def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def readtif(input_file):
    dataset = gdal.Open(input_file)
    print(dataset.GetProjection())
    XSize = dataset.RasterXSize  # 网格的X轴像素数量
    YSize = dataset.RasterYSize  # 网格的Y轴像素数量
    gtf = dataset.GetGeoTransform()  # 投影转换信息
    ProjectionInfo = dataset.GetProjection()  # 投影信息
    
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    
    """
        获取经纬度信息123
        geoTransform[0]是左上角像元的东坐标；

        geoTransform[3]是左上角像元的北坐标；

        geoTransform[1]是影像宽度上的分辨率；

        geoTransform[5]是影像高度上的分辨率；

        geoTransform[2]是旋转, 0表示上面为北方；

        geoTransform[4]是旋转, 0表示上面为北方；

        2相应的放射变换公式：

        Xp = geoTransform [0] +Xpixel*geoTransform [1]+Yline*geoTransform [2];

        Yp = geoTransform [3] + Xpixel*geoTransform [4] + YlineL*geoTransform [

    """
    x_range = range(0, XSize)
    y_range = range(0, YSize)
    x, y = np.meshgrid(x_range, y_range)

    lon = gtf[0] + x * gtf[1] + y * gtf[2]
    lat = gtf[3] + x * gtf[4] + y * gtf[5]
    
    return lon[0,:], lat[:,0], data

if __name__ == '__main__':
    # gdal.AllRegister()
    # dataset = gdal.Open('/home/damon/LSTdownscaled/landsat/LC08_L1TP_120045_20200220_20200822_02_T1_B4.TIF')
    # print('数据投影：')
    # print(dataset.GetProjection())
    # print('数据的大小（行，列）：')
    # print('(%s %s)' % (dataset.RasterYSize, dataset.RasterXSize))

    # x = 316185
    # y = 2513115
    # lon = 122.47242
    # lat = 52.51778
    # row = 2399
    # col = 3751

    # print('投影坐标 -> 经纬度：')
    # coords = geo2lonlat(dataset, x, y)
    # print('(%s, %s)->(%s, %s)' % (x, y, coords[0], coords[1]))
    # print('经纬度 -> 投影坐标：')
    # coords = lonlat2geo(dataset, lon, lat)
    # print('(%s, %s)->(%s, %s)' % (lon, lat, coords[0], coords[1]))

    # print('图上坐标 -> 投影坐标：')
    # coords = imagexy2geo(dataset, row, col)
    # print('(%s, %s)->(%s, %s)' % (row, col, coords[0], coords[1]))
    # print('投影坐标 -> 图上坐标：')
    # coords = geo2imagexy(dataset, x, y)
    # print('(%s, %s)->(%s, %s)' % (x, y, coords[0], coords[1]))
    
    lon, lat, data = readtif(r'D:\xiaojie\data1\final\landlst\20130809_lst.tif')
    # lon1, lat1, data1 = readtif('/home/damon/LSTdownscaled/landsat/test1/Global.tif')

    
    #============================================================================================================
    #4326 为原始空间参考的ESPG编号  2331为目标空间参考的ESPG编号
    dataset = gdal.Open(r'D:\xiaojie\data1\final\landlst\20130809_lst.tif')
    # point2 = lonlat2geo(dataset,24.103209723234617,112.98523662456509)
    point3 = geo2lonlat(dataset,lon[0] ,lat[0])
    # point_trans=point_transform(4326,32650,24.103209723234617,112.98523662456509)
    # point_trans1=point_transform(32650,4326,91756.2661223599,2671500.9063564143) #geo_to_latlon
    # geo = imagexy2geo(dataset, 2, 3)
    
    
    #===========================================================================================
    

    # ==================================================================================================
    # dataset1 = gdal.Open('/home/damon/LSTdownscaled/landsat/test1/Global.tif')
    # row1 = 3565
    # col1 = 1142
    # geo1 = imagexy2geo(dataset1, row1, col1)
    # lonlat1 = geo2lonlat(dataset1, geo1[0], geo1[1])
    # d1 = data1[row1][col1]
    
    # geo = lonlat2geo(dataset, lonlat1[0], lonlat1[1])
    # imagexy = geo2imagexy(dataset, geo[0],geo[1])
    
    # row = int(round(imagexy[1]))
    # col = int(round(imagexy[0])) 
    # d = data[row][col]
    # print(point_trans1)
