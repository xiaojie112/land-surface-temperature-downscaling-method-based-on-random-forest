# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:40:58 2023

@author: a8362
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# fig, axes = plt.subplots(4, 1, figsize=(5, 5))
# fig.subplots_adjust(hspace=4)

# # 第一个colorbar使用线性的Normalize.
# cmap1 = copy.copy(cm.viridis)
# norm1 = mcolors.Normalize(vmin=0, vmax=100)
# im1 = cm.ScalarMappable(norm=norm1, cmap=cmap1)
# cbar1 = fig.colorbar(
#     im1, cax=axes[0], orientation='horizontal',
#     ticks=np.linspace(0, 100, 11),
#     label='colorbar with Normalize'
# )

# # 第二个colorbar开启extend参数.
# cmap2 = copy.copy(cm.viridis)
# cmap2.set_under('black')
# cmap2.set_over('red')
# norm2 = mcolors.Normalize(vmin=0, vmax=100)
# im2 = cm.ScalarMappable(norm=norm2, cmap=cmap2)
# cbar2 = fig.colorbar(
#     im2, cax=axes[1], orientation='horizontal',
#     extend='both', ticks=np.linspace(0, 100, 11),
#     label='extended colorbar with Normalize'
# )

# # 第三个colorbar使用对数的LogNorm.
# cmap3 = copy.copy(cm.viridis)
# norm3 = mcolors.LogNorm(vmin=1E0, vmax=1E3)
# im3 = cm.ScalarMappable(norm=norm3, cmap=cmap3)
# # 使用LogNorm时,colorbar会自动选取合适的Locator和Formatter.
# cbar3 = fig.colorbar(
#     im3, cax=axes[2], orientation='horizontal',
#     label='colorbar with LogNorm',
# )

# # 第四个colorbar使用BoundaryNorm.
# bins = [0, 1, 10, 20, 50, 100]
# nbin = len(bins) - 1
# cmap4 = cm.get_cmap('viridis', nbin)
# norm4 = mcolors.BoundaryNorm(bins, nbin)
# im4 = cm.ScalarMappable(norm=norm4, cmap=cmap4)
# # 使用BoundaryNorm时,colorbar会自动按bins标出刻度.
# cbar4 = fig.colorbar(
#     im4, cax=axes[3], orientation='horizontal',
#     label='colorbar with BoundaryNorm'
# )

# plt.show()


# =============================================

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

def add_box(ax):
    '''用红框标出一个ax的范围.'''
    axpos = ax.get_position()
    rect = mpatches.Rectangle(
        (axpos.x0, axpos.y0), axpos.width, axpos.height,
        lw=3, ls='--', ec='r', fc='none', alpha=0.5,
        transform=ax.figure.transFigure
    )
    ax.add_patch(rect)

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距,width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

def test_data():
    '''生成测试数据.'''
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2) + np.exp(-Y**2)
    # 将Z缩放至[0, 100].
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 100

    return X, Y, Z

# X, Y, Z = test_data()
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# fig.subplots_adjust(hspace=0.2, wspace=0.2)

# # 提前用红框圈出每个ax的范围,并关闭刻度显示.
# for ax in axes.flat:
#     add_box(ax)
#     ax.axis('off')

# # 第一个子图中不画出colorbar.
# im = axes[0, 0].pcolormesh(X, Y, Z, shading='nearest')
# axes[0, 0].set_title('without colorbar')

# # 第二个子图中画出依附于ax的垂直的colorbar.
# im = axes[0, 1].pcolormesh(X, Y, Z, shading='nearest')
# cbar = fig.colorbar(im, ax=axes[0, 1], orientation='vertical')
# axes[0, 1].set_title('add vertical colorbar to ax')

# # 第三个子图中画出依附于ax的水平的colorbar.
# im = axes[1, 0].pcolormesh(X, Y, Z, shading='nearest')
# cbar = fig.colorbar(im, ax=axes[1, 0], orientation='horizontal')
# axes[1, 0].set_title('add horizontal colorbar to ax')

# # 第三个子图中将垂直的colorbar画在cax上.
# im = axes[1, 1].pcolormesh(X, Y, Z, shading='nearest')
# cax = add_right_cax(axes[1, 1], pad=0.02, width=0.02)
# cbar = fig.colorbar(im, cax=cax)
# axes[1, 1].set_title('add vertical colorbar to cax')

# plt.show()

# ==========================================================
X, Y, Z = test_data()
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=100)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# 调节子图间的宽度,以留出放colorbar的空间.
fig.subplots_adjust(wspace=0.4)

for ax in axes.flat:
    ax.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(im, cax=cax)

plt.show()