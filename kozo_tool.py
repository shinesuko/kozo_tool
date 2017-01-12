#coding: utf-8
#! D:\Anaconda3\envs\py27
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import numpy as np
import seaborn as sns; sns.set()
import math
import os
from datetime import datetime


def read_current_data(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass

    if filename:
        print filename.encode('utf-8')
        df = pd.read_table(filename,delimiter='\t',escapechar='&',header=6,index_col=False,parse_dates=[0],na_values='NoData')
        df.index.names = ['index']
        df.columns.names = ['column']
        return [df,filename]
    else:
        print("Canceled!")
        df=[]
        return [df,filename]

def plot_figure(df,filename=[],show=False):
    fig=plt.figure()
    fig.patch.set_facecolor('white')  # 図全体の背景色
    ax=plt.axes()
    plt.hold(True)
    plt.plot(df['DateTime'],df.iloc[:,3],'r',label=list(df)[3])
    plt.plot(df['DateTime'],df.iloc[:,5],'g',label=list(df)[5])
    plt.plot(df['DateTime'],df.iloc[:,7],'b',label=list(df)[7])
    plt.plot(df['DateTime'],df.iloc[:,9],'k',label=list(df)[9])
    plt.ylim([1E-5,1E-1])
    plt.yscale('log')
    plt.legend(loc='lower right',prop={'size':10})
    plt.xlabel('Time')
    plt.ylabel('Current [A]')
    plt.title(filename)
    plt.xticks(rotation=70)
    timeformat = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(timeformat)
    plt.tight_layout()  # タイトルの被りを防ぐ
    if show:
        plt.show()
    else:
        pass
    return [fig,ax]

def read_array_data(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass

    if filename:
        print filename.encode('utf-8')
        # df = pd.read_table(filename,delimiter='\t',index_col=False,names=None)
        # df.index.names = ['index']
        # print len(df.index)
        # df.columns.names = ['column']
        data = np.loadtxt(filename,delimiter="\t")
        return [data,filename]
    else:
        print("Canceled!")
        df=[]
        return [data,filename]

def plot_heatmap(array,title=[],show=False):
    fig=plt.figure()
    ax = sns.heatmap(array,cmap='BuGn')
    sns.plt.title(title)
    if show:
        sns.plt.show()
    return ax

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def save_figure(fig,filename):
    joblib.dump(fig,filename)

def load_figure(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass

    if filename:
        fig=joblib.load(filename)
        fig.show()
    else:
        print 'canceled!'

def save_data(data,filename):
    joblib.dump(data,filename,compress=3)

def load_data(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass

    if filename:
        data=joblib.load(filename)
        return data
    else:
        print 'canceled!'
        return []

class structure():
    pass

def create_master_data(filename=[]):
    print('Starting... create_master_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open master csv.', title=None, default='*', filetypes=['*.csv'], multiple=False)
    else:
        pass

    if os.path.isfile(os.path.dirname(filename)+'/master.mydata'):
        #print
        print(os.path.dirname(filename).encode('utf-8')+'/master.mydata exists.')
        return os.path.dirname(filename)+'/master.mydata'
    else:
        #anonymous class
        data=structure()
        data.version=1.0
        data.field=['version','field']

        #open master data
        print filename.encode('utf-8')
        master = pd.read_table(filename,delimiter=',',escapechar='&',header=0,index_col=False,parse_dates=[9,10,11,12],na_values=['NoData',' ','?'])
        master.index.names = ['index']
        master.columns.names = ['column']
        data.master=master
        data.field.append('master')

        #timestamp
        data.timestamp=datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        data.field.append('timestamp')

        #save master.mydata
        save_data(data,os.path.dirname(filename)+'/master.mydata')
        print('Saving... '+os.path.dirname(filename).encode('utf-8')+'/master.mydata')
        return os.path.dirname(filename)+'/master.mydata'

    print('End of create_master_data.')

def create_irradiation_data(filename=[]):
    print('Starting... create_irradiation_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open master mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        data=load_data(filename)
        data.field.append('irradiation')
        for i, v in data.master.iterrows():
            print(i, v['ion'], v['number'])
            #irradiation
            data.irradiation=v
            formatted = '%03d' % int(v['number'])
            #save data
            save_data(data,os.path.dirname(filename)+'/'+v['ion']+formatted+'.mydata')
            print('Saving... irradiation data at'+os.path.dirname(filename).encode('utf-8')+'/'+v['ion']+formatted+'.mydata')
    else:
        pass

    print('End of create_irradiation_data.')

def add_current_data(filename=[]):
    print('Starting... add_current_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        os.chdir(os.path.dirname(filename))
        os.chdir("..")
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        ion=data.irradiation['ion']
        print(ion+formatted+'.mydata is seleced')
        filenames_current=easygui.fileopenbox(msg='Open '+ion+formatted+' current data.', title=None, default='*', filetypes=['*.txt'], multiple=True)
        if filenames_current:
            data.current=pd.DataFrame(index=[], columns=['DateTime'])
            data.field.append('current')
            print(data.current)
            for file in filenames_current:
                df,_=read_current_data(filename=file)
                data.current=pd.merge(data.current,df,on='DateTime',how='outer')
            print(data.current)
            #save current data
            save_data(data,filename)
            print('Saving... current data at '+filename.encode('utf-8'))
        else:
            print('Canceled.')

    else:
        pass
    print('End of add_current_data.')

def add_SEE_data(filename=[]):
    print('Starting... add_SEE_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        os.chdir(os.path.dirname(filename))
        os.chdir("..")
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        ion=data.irradiation['ion']
        print(ion+formatted+'.mydata is seleced')
        filenames_SEE=easygui.fileopenbox(msg='Open '+ion+formatted+' SEE data.', title=None, default='*', filetypes=['*.jpg'], multiple=True)
        if filenames_SEE:
            data.SEE=pd.DataFrame(index=[], columns=['DateTime'])
            data.field.append('SEE')
            for file in filenames_SEE:
                name,_ = os.path.splitext(os.path.basename(file).encode('utf-8'))
                date=pd.to_datetime(name[:15],format='%Y%m%d_%H%M%S')
                data.SEE=pd.concat([data.SEE, pd.DataFrame(date,index=[0], columns=['DateTime'])], ignore_index=True)
            print data.SEE
            #save SEE data
            save_data(data,filename)
            print('Saving... SEE data at '+filename.encode('utf-8'))

        else:
            print('Canceled.')
    else:
        pass
    print('End of add_SEE_data.')
