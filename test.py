#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import pandas as pd
import numpy as np


data=kt.structure()
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6,4),index = dates, columns = list("ABCD"))
data.master=df

print data.master
print type(data)
print type(data.master)

kt.save_data(data,filename=u'C:/Users/14026/Desktop/test.mydata')

print('---------------------------------')
data=kt.load_data()
print(type(data.master))
print (data.master)
