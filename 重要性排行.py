# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:25:48 2023

@author: a8362
"""

import numpy as np
import matplotlib.pyplot as plt

# 准备数据
# 300m mndwi elevation ndvi ndmi msavi ndbi savi

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
labels = ['MNDWI','Elevation','NDVI','NDMI','MSAVI','NDBI','SAVI']
fig,ax = plt.subplots(2,2,figsize=(14,10))









fbl300 = [0.12, 0.21, 0.27, 0.15, 0.06, 0.14, 0.05]
fbl500 = [0.13, 0.2, 0.24, 0.16, 0.09, 0.12, 0.06]
fbl1000 = [0.15, 0.2, 0.21, 0.17, 0.09, 0.11, 0.07]
fbl2000 = [0.12, 0.23, 0.25, 0.16, 0.08, 0.09, 0.07]
fbl5000 = [0.15, 0.19, 0.22, 0.17, 0.11, 0.08, 0.08]
# 转换x轴数据为整数类型
x = np.arange(len(fbl300))
# 绘制多个条形统计图
width = 0.07
# fig, ax = plt.subplots()
ax[0][0].bar(x - 2 * width, fbl300, width, label='300m')
ax[0][0].bar(x - width, fbl500, width, label='500m')
ax[0][0].bar(x, fbl1000, width, label='1000m')
ax[0][0].bar(x + width, fbl2000, width, label='2000m')
ax[0][0].bar(x + width * 2, fbl5000, width, label='5000m')
# 添加图例、标签和标题
ax[0][0].set_xticks(x)
ax[0][0].set_xticklabels(labels)
ax[0][0].set_ylabel('重要性')
ax[0][0].set_title('春季随机森林模型地表参数重要性')
ax[0][0].legend()#设置图例





# In[]
summer300 = [0.14, 0.19, 0.26, 0.11, 0.08, 0.17, 0.05]
summer500 = [0.17, 0.23, 0.25, 0.13, 0.05, 0.18, 0.03]
summer1000 = [0.14, 0.22, 0.23, 0.14, 0.07, 0.16, 0.05]
summer2000 = [0.14, 0.22, 0.27, 0.1, 0.06, 0.15, 0.06]
summer5000 = [0.16, 0.21, 0.24, 0.15, 0.07, 0.13, 0.04]
ax[0][1].bar(x - 2 * width, summer300, width, label='300m')
ax[0][1].bar(x - width, summer500, width, label='500m')
ax[0][1].bar(x, summer1000, width, label='1000m')
ax[0][1].bar(x + width, summer2000, width, label='2000m')
ax[0][1].bar(x + width * 2, summer5000, width, label='5000m')
# 添加图例、标签和标题
ax[0][1].set_xticks(x)
ax[0][1].set_xticklabels(labels)
ax[0][1].set_ylabel('重要性')
ax[0][1].set_title('夏季随机森林模型地表参数重要性')
ax[0][1].legend()#设置图例


# In[]
autumn300 = [0.10, 0.25, 0.22, 0.18, 0.07, 0.13, 0.05]
autumn500 = [0.16, 0.27, 0.25, 0.19, 0.07, 0.1, 0.04]
autumn1000 = [0.14, 0.24, 0.23, 0.18, 0.08, 0.08, 0.07]
autumn2000 = [0.13, 0.24, 0.24, 0.19, 0.05, 0.06, 0.03]
autumn5000 = [0.13, 0.22, 0.21, 0.18, 0.06, 0.04, 0.05]
ax[1][0].bar(x - 2 * width, autumn300, width, label='300m')
ax[1][0].bar(x - width, autumn500, width, label='500m')
ax[1][0].bar(x, autumn1000, width, label='1000m')
ax[1][0].bar(x + width, autumn2000, width, label='2000m')
ax[1][0].bar(x + width * 2, autumn5000, width, label='5000m')
# 添加图例、标签和标题
ax[1][0].set_xticks(x)
ax[1][0].set_xticklabels(labels)
ax[1][0].set_ylabel('重要性')
ax[1][0].set_title('秋季随机森林模型地表参数重要性')
ax[1][0].legend()#设置图例




# In[]
winter300 = [0.12, 0.24, 0.21, 0.14, 0.05, 0.18, 0.06]
winter500 = [0.16, 0.24, 0.23, 0.13, 0.05, 0.15, 0.04]
winter1000 = [0.15, 0.27, 0.21, 0.15, 0.06, 0.12, 0.03]
winter2000 = [0.09, 0.26, 0.2, 0.12, 0.09, 0.1, 0.07]
winter5000 = [0.1, 0.23, 0.22, 0.16, 0.07, 0.08, 0.05]
ax[1][1].bar(x - 2 * width, winter300, width, label='300m')
ax[1][1].bar(x - width, winter500, width, label='500m')
ax[1][1].bar(x, winter1000, width, label='1000m')
ax[1][1].bar(x + width, winter2000, width, label='2000m')
ax[1][1].bar(x + width * 2, winter5000, width, label='5000m')
# 添加图例、标签和标题
ax[1][1].set_xticks(x)
ax[1][1].set_xticklabels(labels)
ax[1][1].set_ylabel('重要性')
ax[1][1].set_title('冬季随机森林模型地表参数重要性')
ax[1][1].legend()#设置图例


# 显示图形
plt.show()
