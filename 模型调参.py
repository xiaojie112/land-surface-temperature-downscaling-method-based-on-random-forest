# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:17:09 2023

@author: a8362
"""

# 参考链接：https://zhuanlan.zhihu.com/p/126288078
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[]
# 导入乳腺癌数据集
data = load_breast_cancer()

# 建立随机森林
rfc = RandomForestClassifier(n_estimators=100, random_state=90)

score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
score_pre

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
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
for i in range(60,80):
    rfc = RandomForestClassifier(n_estimators=i
                                ,random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)+60))

# 绘制学习曲线
x = np.arange(60,80)
plt.subplot(111)
plt.plot(x, score_lt,'o-')
plt.show()


# In[]
# 建立n_estimators为45的随机森林
rfc = RandomForestClassifier(n_estimators=73, random_state=90)

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)


# In[]
# 用网格搜索调整max_features
param_grid = {'max_features':np.arange(5,31)}

rfc = RandomForestClassifier(n_estimators=73
                            ,random_state=90
                            ,max_depth=8)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)  