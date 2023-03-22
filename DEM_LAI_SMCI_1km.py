from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from configargparse import ArgParser
from statistics import mean
import pandas as pd
import os
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib as mpl
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

newcmap = truncate_colormap(plt.cm.terrain, 0.23, 1, 128)
fig = plt.figure(figsize=(10, 10),dpi=200)   #画布大小
config = {"font.family": 'Times New Roman',
          "font.size": 12}
rcParams.update(config)



def main(China_site_dir,name):
    China_lon = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/longitude.npy")
    China_lat = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/latitude.npy")  ##  1km 的经纬度
    file1 = name + '.npy'
    observed_path = os.path.join(China_site_dir, file1)
    observed_sm = np.load(observed_path)
    observed_sm = np.squeeze(observed_sm)
    mask = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/mask.npy")
    observed_SM_values = observed_sm
    observed_SM_values[mask==0]=-9999
    m = Basemap(llcrnrlon=np.min(China_lon),
                llcrnrlat=np.min(China_lat),
                urcrnrlon=np.max(China_lon),
                urcrnrlat=np.max(China_lat),
               )
    m.readshapefile(r"C:\Users\Admin\Desktop\222\basemap\china-shapefiles/china",  'china', drawbounds=True)
    parallels = np.arange(0.,81,10.)
    m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
    meridians = np.arange(10.,351.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])

    lon, lat = np.meshgrid(China_lon, China_lat)
    xi, yi = m(lon, lat)
    if name == "DEM_1km":
        norm3 = mpl.colors.BoundaryNorm(np.arange(0,7600,100), newcmap.N)  # 标准化level，映射色标
        cs = m.contourf(xi,yi, observed_SM_values, np.arange(0, 7600, 100), norm= norm3,cmap=newcmap)
        cbar = m.colorbar(cs, location='bottom', pad="15%")  ##  vertical  bottom
        cbar.set_label('m')
        plt.title("DEM")
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE\DEM_LAI/DEM.png")
        plt.show()
    elif name == "lai_1km_2016_oneday":
        norm3 = mpl.colors.BoundaryNorm(np.arange(0, 7, 0.05), newcmap.N)  # 标准化level，映射色标
        observed_SM_values = np.maximum(observed_SM_values,0.002)
        observed_SM_values[mask == 0] = -9999
        cs = m.contourf(xi,yi, observed_SM_values, np.arange(0, 7, 0.05), norm= norm3,cmap=newcmap)
        cbar = m.colorbar(cs, location='bottom', pad="15%")  ##  vertical  bottom
        cbar.set_label('m$^{2}$/m$^{2}$')
        plt.title("LAI")
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE\DEM_LAI/LAI.png")
        plt.show()


"""
高程 DEM 的图  和。叶面积指数 LAI 
lai_1km_2016_oneday
DEM_1km
"""
if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--China_site_dir', type=str, default=r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/", help='Path to data')# '../10_layer/total_15_ge_var_10cm/'
    p.add_argument('--name', type=str, default='lai_1km_2016_oneday', help='name for prediction model')
    args = p.parse_args()

    main(
        China_site_dir=args.China_site_dir,
        name=args.name
    )



def Found_lat(station_la,lat):  ##  找到 跟站点位置  最近的纬度
    dif_lat = float("inf")
    for la_index in range(len(lat)):
        abs_data = abs(lat[la_index] - station_la)
        if abs_data < dif_lat:
            lat_index = la_index
            dif_lat = abs_data
    return lat_index

def Found_lon(station_lo,lon):
    dif_lon = float("inf")
    for lo_index in range(len(lon)):
        abs_data = abs(lon[lo_index] - station_lo)
        if abs_data < dif_lon:
            lon_index = lo_index
            dif_lon = abs_data
    return lon_index

def ju_bu_tu_1(Product_Name,station_lat,station_lon):
    China_lon = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/longitude.npy")
    China_lat = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/latitude.npy")  ##  1km 的经纬度
    horizontal_axis_bias = 200
    vertical_axis_bias = 300
    config = {"font.family": 'Times New Roman',
              "font.size": 12}
    rcParams.update(config)

    lat_index = Found_lat(station_lat,China_lat)
    lon_index = Found_lon(station_lon,China_lon)

    observed_sm = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/{}.npy".format(Product_Name))## 5320, 7560的shape
    mask = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/mask.npy")
    observed_sm[mask==0] = 999999
    # observed_sm = observed_sm * mask
    observed_sm = np.squeeze(observed_sm)
    # observed_sm = np.maximum(observed_sm, 0.002)

    observed_SM_values = observed_sm[(lat_index - vertical_axis_bias):(lat_index + vertical_axis_bias), (lon_index - horizontal_axis_bias):(lon_index + horizontal_axis_bias)]
    # observed_SM_values[np.where(observed_SM_values<0.1)]=-9999

    aa_lat = China_lat[(lat_index - vertical_axis_bias):(lat_index + vertical_axis_bias)]
    bb_lon = China_lon[(lon_index - horizontal_axis_bias):(lon_index + horizontal_axis_bias)]
    m = Basemap(llcrnrlon=np.min(bb_lon),
                llcrnrlat=np.min(aa_lat),
                urcrnrlon=np.max(bb_lon),
                urcrnrlat=np.max(aa_lat))
    lon, lat = np.meshgrid(bb_lon, aa_lat)
    xi, yi = m(lon, lat)
    m.readshapefile(r"C:\Users\Admin\Desktop\222\basemap\china-shapefiles/china", 'china', drawbounds=True)
    parallels = np.arange(0., 81.,0.5)
    meridians = np.arange(10., 351., 0.5)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    Elevation_max = np.max(observed_SM_values)
    # norm3 = mpl.colors.BoundaryNorm(np.arange(0,Elevation_max,int(Elevation_max/10)), newcmap.N)  # 标准化level，映射色标

    # cs = m.contourf(xi,yi, observed_SM_values, np.arange(0, Elevation_max, 100),norm=norm3,cmap=newcmap)

    if Product_Name == "DEM_1km":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap=newcmap, marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("DEM")
        cbar.set_label('m')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/last/DEM.png")
        plt.show()
    elif Product_Name == "lai_1km_2016_oneday":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap=newcmap, marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("LAI")
        cbar.set_label('m$^{2}$/m$^{2}$')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/last/LAI.png")
        plt.show()
    elif Product_Name == "SMCI_30cm":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap="YlGnBu", marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("SMCI1.0")
        cbar.set_label('m$^{3}$/m$^{3}$')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/last/SMCI.png")
        plt.show()


# ju_bu_tu_1(Product_Name="lai_1km_2016_oneday",station_lat = 30,station_lon =103)

"""
Product_Name  从以下三个中选择：
SMCI_30cm
DEM_1km
lai_1km_2016_oneday
"""



"""
以下是 将局部图 再次放大：小小图
"""
def xiao_xiao_tu(Product_Name,station_lat,station_lon):
    China_lon = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/longitude.npy")
    China_lat = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/latitude.npy")  ##  1km 的经纬度

    horizontal_axis_bias = 50
    vertical_axis_bias = 50
    config = {"font.family": 'Times New Roman',
              "font.size": 12}
    rcParams.update(config)

    lat_index = Found_lat(station_lat,China_lat)
    lon_index = Found_lon(station_lon,China_lon)

    observed_sm = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/{}.npy".format(Product_Name))## 5320, 7560的shape
    mask = np.load(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560/mask.npy")
    observed_sm[mask==0] = 999999
    # observed_sm = observed_sm * mask
    observed_sm = np.squeeze(observed_sm)
    # observed_sm = np.maximum(observed_sm, 0.002)

    observed_SM_values = observed_sm[(lat_index - vertical_axis_bias-10):(lat_index + vertical_axis_bias), (lon_index - horizontal_axis_bias+5):(lon_index + horizontal_axis_bias+7)]
    # observed_SM_values[np.where(observed_SM_values<0.1)]=-9999

    aa_lat = China_lat[(lat_index - vertical_axis_bias-10):(lat_index + vertical_axis_bias)]
    bb_lon = China_lon[(lon_index - horizontal_axis_bias+5):(lon_index + horizontal_axis_bias+7)]
    m = Basemap(llcrnrlon=np.min(bb_lon),
                llcrnrlat=np.min(aa_lat),
                urcrnrlon=np.max(bb_lon),
                urcrnrlat=np.max(aa_lat))
    lon, lat = np.meshgrid(bb_lon, aa_lat)
    xi, yi = m(lon, lat)
    m.readshapefile(r"C:\Users\Admin\Desktop\222\basemap\china-shapefiles/china", 'china', drawbounds=True)
    parallels = np.arange(0., 81, 0.1)
    meridians = np.arange(10., 351., 0.1)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])

    if Product_Name == "DEM_1km":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap=newcmap, marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("DEM")
        cbar.set_label('m')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/xiao_xiao_tu/DEM.png")
        plt.show()
    elif Product_Name == "lai_1km_2016_oneday":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap=newcmap, marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("LAI")
        cbar.set_label('m$^{2}$/m$^{2}$')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/xiao_xiao_tu/LAI.png")
        plt.show()
    elif Product_Name == "SMCI_30cm":
        cs = m.scatter(xi, yi, s=100, c=observed_SM_values, cmap="YlGnBu", marker="s")
        cbar = m.colorbar(cs, location='right', pad="15%")
        plt.title("SMCI1.0")
        cbar.set_label('m$^{3}$/m$^{3}$')
        plt.savefig(r"C:\Users\Admin\Desktop\Temp_Data1\2016one_day_9km\SMCI_1km_4320_7560\PICTURE/xiao_xiao_tu/SMCI.png")
        plt.show()

# xiao_xiao_tu(Product_Name="DEM_1km",station_lat = 31.7,station_lon =103.7)

"""
SMCI_30cm
DEM_1km
lai_1km_2016_oneday
"""



