#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import easygui
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()

filenames=easygui.fileopenbox(msg=None, title=None, default='*', filetypes=['*.txt'], multiple=True)

length=len(filenames)#file数
struct=[]#初期化

for filename in filenames:
    data,filename=kt.read_array_data(filename=filename)
    struct.append(data)

coef=kt.corr2(struct[0],struct[1])
print coef.astype('str')

# 新規のウィンドウを描画
fig = plt.figure()
# サブプロットを追加
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
sns.heatmap(struct[0],cmap='BuGn',ax=ax1)
sns.heatmap(struct[1],cmap='BuGn',ax=ax2)
plt.table(cellText=coef.astype('str'),colLabels='Corros Coef',loc='center')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
plt.tight_layout()
plt.show()
