#coding: utf-8
#! D:\Anaconda3\envs\py27
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import joblib
import numpy as np
# import seaborn as sns; sns.set()
import math
import os
from datetime import datetime
import re
import mpld3
import dill
from scipy import signal
import zipfile
from scipy import interpolate
import netCDF4
import tarfile
from PIL import Image
from scipy.stats import linregress
from scipy import stats
import lmfit
from tqdm import tqdm,tqdm_notebook

def read_current_data(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass
    if filename:
        print filename.encode('utf-8')
        try:
            df = pd.read_table(filename,delimiter='\t',escapechar='&',header=1,skipfooter=1,index_col=False,parse_dates=[0],na_values='NoData',engine='python')
            print(len(df.columns))
            if len(df.columns)==6:
                df.columns = ['DateTime','Elap[sec]','Ch1[I]','unknown1','Ch2[I]','unknown2']
            else:
                df.columns = ['DateTime','Elap[sec]','Ch1[I]','Ch2[I]']
            print('reading as 6581 file format.')
        except:
            print('reading as 6629 file format.')
            df = pd.read_table(filename,delimiter='\t',escapechar='&',header=6,index_col=False,parse_dates=[0],na_values='NoData')
        df.index.names = ['index']
        df.columns.names = ['column']
        return [df,filename]
    else:
        print('Canceled!')
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
        data = np.loadtxt(filename,delimiter='\t')
        return [data,filename]
    else:
        print('Canceled!')
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
        master = pd.read_table(filename,delimiter=',',escapechar='&',header=0,index_col=False,parse_dates=[9,10,11,12,13,14],na_values=['NoData',' ','?','#DIV/0!'])
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
            print(i, v['ion'], str(v['number']))
            #irradiation
            # data.irradiation=v
            data.irradiation=data.master[(data.master.ion==v['ion']) & (data.master.number==v['number'])]
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
        os.chdir('..')
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        try:
            ion=data.irradiation.iloc[0]['ion']
        except:
            ion=data.irradiation.ion
        # print(data.irradiation['ion'].values)
        # print(data.irradiation.iloc[0]['ion'])
        # print(data.irradiation.at[0,'ion'])
        print(ion+formatted+'.mydata is seleced')
        filenames_current=easygui.fileopenbox(msg='Open '+ion+formatted+' current data.', title=None, default='*', filetypes=['*.txt'], multiple=True)
        if filenames_current:
            data.current=pd.DataFrame(index=[], columns=['DateTime'])
            if 'current' in data.field:
                pass
            else:
                data.field.append('current')
            print(data.current)
            for file in filenames_current:
                df,_=read_current_data(filename=file)
                print df
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
        os.chdir('..')
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        ion=data.irradiation['ion'].values
        print(ion+formatted+'.mydata is seleced')
        filenames_SEE=easygui.fileopenbox(msg='Open '+ion+formatted+' SEE data.', title=None, default='*', filetypes=['*.png','*.jpg'], multiple=True)
        if filenames_SEE:
            data.SEE=pd.DataFrame(index=[], columns=['DateTime'])
            if 'SEE' in data.field:
                pass
            else:
                data.field.append('SEE')
            for file in filenames_SEE:
                name,_ = os.path.splitext(os.path.basename(file).encode('utf-8'))
                try:
                    date=pd.to_datetime(name[:15],format='%Y%m%d_%H%M%S')
                except:
                    date=pd.to_datetime(name[:14],format='%Y%m%d%H%M%S')
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

def add_figure(filename=[],show=False,html=False,png=False,eps=False,add_data=False,fix_time=False,ymin=1E-6,ymax=1E-1,yscale='log',edge_detection=False):
    print('Starting... add_figure.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata to add figure.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        data=load_data(filename=filename)
        print('Reading... '+filename.encode('utf-8'))
        if  'current' in dir(data):
            print('Reading... data.current')
            column_names=data.current.columns #data.curren dataframeの列名list取得
            current_list=[x for x in column_names if re.search('\[I\]', x) ]#column_names中の[I]を含む列を取得
            print current_list
            #create figure
            # fig=plt.figure()
            # fig.patch.set_facecolor('white')  # 図全体の背景色
            fig=plt.figure(facecolor ="#FFFFFF")
            plt.style.use('classic')
            ax=plt.axes()
            # plt.hold(True)
            current_list=[x for x in column_names if re.search('\[I\]', x) ] #name list to plot
            for i,name in enumerate(current_list):
                plt.plot(data.current['DateTime'],data.current[name],color=cm.jet(float(i) / len(current_list)),label=name[0:-3])
            #add irradiation duration
            # print(data.irradiation.irradiation_start)
            # print(data.irradiation.irradiation_end)

            #calc difference
            # diff=data.current['DateTime'][0]-data.irradiation.irradiation_start
            # print(diff)
            # plt.hold(True)
            # plt.axvspan(ymin=1E-5,ymax=1,xmin=data.irradiation.irradiation_start+diff, xmax=data.irradiation.irradiation_end+diff,facecolor='gray', alpha=0.5)

            plt.ylim([ymin,ymax])
            plt.yscale(yscale)
            plt.legend(loc='lower right',prop={'size':10})
            plt.xlabel('Time')
            plt.ylabel('Current [A]')
            plt.title(os.path.basename(filename).encode('utf-8'))
            plt.xticks(rotation=30)
            timeformat = mdates.DateFormatter('%H:%M:%S')
            ax.xaxis.set_major_formatter(timeformat)
            plt.tight_layout()  # タイトルの被りを防ぐ

            if edge_detection:
                for i,name in enumerate(current_list):
                    index=edge_detect(data.current[name],threshold=0.0000005)
                    plt.plot(data.current.DateTime[index],data.current.loc[index,name],color=cm.jet(float(i) / len(current_list)),marker='^',linestyle = 'None',markerfacecolor='None')
            else:
                pass

            #save as html
            if html:
                save_filename = open(filename[0:-7]+'.html', 'wb')
                mpld3.save_html(fig,save_filename)
                print('Saving... .html format.')
                save_filename.close()
            else:
                pass

            #add_SEE_data
            if add_data:
                #check current time
                if  'current_start' in list(data.master.columns): #instead of dir(data.irradiation)
                    print('Current start data found.')
                    # diff_current_irrad=data.irradiation.current_start.iloc[0]-data.irradiation.irradiation_start.iloc[0]
                    # print(diff_current_irrad)
                    # plt.hold(True)
                    if fix_time:
                        try:
                            plt.axvspan(xmin=data.irradiation.iloc[0]['current_start'],xmax=data.irradiation.iloc[0]['current_end'],facecolor='gray', alpha=0.1)
                        except:
                            plt.axvspan(xmin=data.irradiation.current_start.iloc[0],xmax=data.irradiation.current_end.iloc[0],facecolor='gray', alpha=0.1)
                    else:
                        print('fix_time:disabled')
                        try:
                            plt.axvspan(xmin=data.irradiation.iloc[0]['irradiation_start'],xmax=data.irradiation.iloc[0]['irradiation_end'],facecolor='gray', alpha=0.1)
                        except:
                            plt.axvspan(xmin=data.irradiation.irradiation_start,xmax=data.irradiation.irradiation_end,facecolor='gray', alpha=0.1)
                elif  'irradiation_start' in list(data.master.columns): #instead of dir(data.irradiation)
                    try:
                        plt.axvspan(xmin=data.irradiation.iloc[0]['irradiation_start'],xmax=data.irradiation.iloc[0]['irradiation_end'],facecolor='gray', alpha=0.1)
                    except:
                        plt.axvspan(xmin=data.irradiation.irradiation_start,xmax=data.irradiation.irradiation_end,facecolor='gray', alpha=0.1)
                else:
                    # plt.hold(True)
                    #plt.axvspan(data.irradiation.irradiation_start+pd.Timedelta(seconds=16),data.irradiation.irradiation_end+pd.Timedelta(seconds=16),facecolor='gray', alpha=0.1)
                    print('No Current start data.')

                #check SEE
                if  ('SEE' in dir(data)) &('current_start' in list(data.master.columns)):
                    print('SEE data found.')
                    #calc difference
                    # diff=data.irradiation.SEE_start.iloc[0]-data.irradiation.current_start.iloc[0]
                    diff=data.irradiation.SEE_start.iloc[0]-data.irradiation.current_start.iloc[0]
                    if fix_time:
                        if diff < pd.Timedelta(seconds=-1):
                            diff=data.irradiation.current_start.iloc[0]-data.irradiation.SEEstart.iloc[0]
                            print(diff)
                            for t in data.SEE['DateTime']:
                                plt.hold(True)
                                plt.scatter(t+diff,0.01,marker='o')
                        else:

                            for t in data.SEE['DateTime']:
                                # print t,type(t)
                                print t-diff
                                plt.hold(True)
                                plt.scatter(t-diff,0.01,marker='o')
                    else:
                        for t in data.SEE['DateTime']:
                            plt.hold(True)
                            plt.scatter(t,0.01,marker='o')
                elif  'SEE' in dir(data):
                    for t in data.SEE['DateTime']:
                        # print t,type(t)
                        # print t-diff
                        # plt.hold(True)
                        plt.scatter(t,0.01,color='blue',marker='o')

                else:
                    print('No SEE data.')
            else:
                pass

            #save figure as png
            if png:
                fig.savefig(filename[0:-7]+'.png',format='png',dpi=300)
                print('Saving... .png format.')
            else:
                pass


            #save as eps
            if eps:
                fig.savefig(filename[0:-7]+'.eps',format='eps',dpi=300)
                print('Saving... .eps format.')
            else:
                pass

            #plot show
            if show:
                plt.show()
            else:
                plt.clf()
                pass

            # data.fig=fig
            # data.plt=plt
            # data.ax=ax
            # with open('C:/Users/14026/Desktop/test.mydata','wb') as f:
            #     dill.dump(data,f)

        else:
            print(filename.encode('utf-8')+' has no data to plot figure')
    else:
        pass
    print('End of add_figure.')

def add_figure_w_SET(filename=[],show=False,html=False,png=False,eps=False,add_data=False,fix_time=False,ymin=1E-6,ymax=1E-1,yscale='log',edge_detection=False):
    print('Starting... add_figure.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata to add figure.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass
    if filename:
        data=load_data(filename=filename)
        print('Reading... '+filename.encode('utf-8'))
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        # fig=plt.figure(facecolor ="#FFFFFF")
        plt.style.use('classic')

        ax1=fig.add_axes((0,0,1,0.75))
        ax2=fig.add_axes((0,0.8,1,0.2),sharex=ax1)
        # ax1.plot(data.current.DateTime,data.current['VDD33[I]'],label='VDD33')
        # ax1.plot(data.current.DateTime,data.current['VDD18[I]'],label='VDD18')
        # ax1.plot(data.current.DateTime,data.current['VDD[I]'],label='VDD')
        ax1.plot(data.current.sort_values(by=["DateTime"], ascending=True).DateTime,data.current.sort_values(by=["DateTime"], ascending=True)['VDD33[I]'],label='VDD33')
        ax1.plot(data.current.sort_values(by=["DateTime"], ascending=True).DateTime,data.current.sort_values(by=["DateTime"], ascending=True)['VDD18[I]'],label='VDD18')
        ax1.plot(data.current.sort_values(by=["DateTime"], ascending=True).DateTime,data.current.sort_values(by=["DateTime"], ascending=True)['VDD[I]'],label='VDD')
        if 'SET' in dir(data):
            for i in range(1,9,1):
                for time in data.SET.time[data.SET.error=='SMUX_OUT'+str(i)]:
                    ax2.scatter(time,i)
            for time in data.SET.time[data.SET.error=='other']:
                    ax2.scatter(time,9)
        else:
            pass

        ax1.axvspan(xmin=data.irradiation.irradiation_start.values[0],xmax=data.irradiation.irradiation_end.values[0],alpha=0.1)

        ax1.set_yscale('log')
        ax1.set_ylim([0.001,0.1])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.grid()
        ax2.tick_params(labelsize=8,labelbottom='off')
        ax2.grid()
        ax2.set_yticks(range(1,10,1))
        ax2.set_yticklabels(labels=['SMUX_OUT1','SMUX_OUT2','SMUX_OUT3','SMUX_OUT4','SMUX_OUT5','SMUX_OUT6','SMUX_OUT7','SMUX_OUT8','other'])
        # ax2.tick_params(labelleft='off')
        ax2.set_ylim([0,10])
        ax1.legend(loc=4)
        plt.title(data.irradiation.ion.values[0]+'{0:03d}'.format(data.irradiation.number.values[0]))
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Current [A]')
        # plt.tight_layout()  # タイトルの被りを防ぐ
        #save figure as png
        #     plt.tight_layout()
        if show:
            plt.show()
        else:
            pass

        fig.savefig(filename[0:-7]+'.png',format='png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        # fig.savefig(filename[0:-7]+'.png',format='png')
        plt.clf()
        print('Saving... .png format.')
        print('End of add_figure_w_SET.')
    else:
        pass

def update_irradiation_data(filename=[]):
    print('Starting... update_irradiation_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata to update_irradiation_data.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        data=load_data(filename=filename)
        print('Reading... '+filename.encode('utf-8'))
        if os.path.isfile(os.path.dirname(filename)+'/master.mydata'):
            #print
            print(os.path.dirname(filename).encode('utf-8')+'/master.mydata exists.')
            master_data=load_data(filename=os.path.dirname(filename)+'/master.mydata')
            #update master data
            data.master=master_data.master
            #update irradiation data
            try:
                print(data.irradiation.iloc[0]['ion'], data.irradiation.iloc[0]['number'])
                data.irradiation=data.master[(data.master['ion']==data.irradiation.iloc[0]['ion']) & (data.master['number']==data.irradiation.iloc[0]['number'])]
            except:
                                print(data.irradiation['ion'], data.irradiation['number'])
                                data.irradiation=data.master[(data.master['ion']==data.irradiation['ion']) & (data.master['number']==data.irradiation['number'])]
            # print data.irradiation
            print(filename+' has been overwritten.')
            formatted = '%03d' % int(data.irradiation['number'])
            #save data
            save_data(data,filename)
            print('Saving... irradiation data at'+filename)
        else:
            pass
    else:
        pass

    print('End of... update_irradiation_data.')

def update_all_irradiation_data():
    filenames=easygui.fileopenbox(msg='Open .mydata to update irradiation data.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    if filenames:
        for filename in filenames:
            update_irradiation_data(filename=filename)
    else:
        print('Canceled.')

def data2figure(filename=[],show=False,html=True,png=True,eps=False,add_data=True,fix_time=True,ymin=1E-6,ymax=1E-1,yscale='log',SET=True):
    print('Starting... data2figure.')
    filenames=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    if filenames:
        for filename in filenames:
            if SET:
                add_figure_w_SET(filename=filename,show=show)
            else:
                add_figure(filename=filename,show=show,html=html,png=png,eps=eps,add_data=add_data,fix_time=fix_time,ymin=ymin,ymax=ymax,yscale=yscale)
    else:
        print('Canceled.')
    print('End of... data2figure.')

def analyse_SEE_interval():
    print('START')
    filenames=easygui.fileopenbox(msg='Open mydata to analyse data.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    interval=pd.DataFrame(index=[], columns=['DateTime'])
    if filenames:
        for filename in filenames:
            data=load_data(filename=filename)
            if  'SEE' in dir(data):
                # print filename.encode('utf-8')
                tmp=data.SEE.sort_values(by=['DateTime'], ascending=True).diff().astype('timedelta64[s]').dropna()
                interval=pd.concat([interval, tmp], ignore_index=True)
            else:
                print(filename.encode('utf-8')+' structure instance has no attribute SEE')

    else:
        pass
    print interval.describe()
    ax=interval.plot(y=['DateTime'],bins=50,alpha=0.5,kind='hist')
    ax.set_xlabel('Interval [s]')
    ax.legend_.remove()
    fig = plt.gcf()
    fig.savefig(os.path.dirname(filenames[0])+'/interval.png',format='png',dpi=300)
    print('interval figure has been saved at '+os.path.dirname(filenames[0]).encode('utf-8')+'\interval.png')

    stats=structure()
    stats.interval=interval
    stats.filenames=filenames
    stats.version=1.0
    stats.field=['version','field']
    stats.field.append('interval')
    stats.field.append('filenames')
    save_data(stats,os.path.dirname(filenames[0])+'/stats.mydata')
    print('stats data has been saved at '+os.path.dirname(filenames[0]).encode('utf-8')+'\stats.mydata')

    print('END')

def edge_detect(data,threshold=0.000001):
    #difference
    diff=np.diff(data)
    #detect change when 0.000001 A/s
    diff[np.abs(diff)< threshold]=0
    pos_peak=diff.copy()
    neg_peak=diff.copy()
    pos_peak[pos_peak < 0]=0
    neg_peak[neg_peak > 0]=0
    positive_index = signal.find_peaks_cwt(pos_peak,np.arange(1,10,0.1)/10)
    negative_index = signal.find_peaks_cwt(-neg_peak,np.arange(3,5,0.1)/10)
    index=sorted(list(set(negative_index)),key=negative_index.index)
    index.extend(positive_index)
    return index

def get_cur_list_from_zip():
    filename=easygui.fileopenbox(msg='Open .zip to get cur lsit.', title=None, default='*', filetypes=['*.zip'], multiple=False)
    names=list()
    if filename:
        with zipfile.ZipFile(filename, 'r') as zf:
            for name in zf.namelist():
                if '.cur' in name:
                    print(name)
                    names.append(name)
            return [names,filename]
    else:
        return []

def open_cur_in_zip(filename=[],cur=[]):
    if filename:
        if 'alpha.cur' not in cur:
            print('cur file found.')
            with zipfile.ZipFile(filename, 'r') as zf:
                tmp=pd.read_table(zf.open(cur),sep=' ',header=54)
                temp=tmp.drop([tmp.columns[0],tmp.columns[13],tmp.columns[14],tmp.columns[15],tmp.columns[16],tmp.columns[17]],axis=1)
                temp.columns=['Vsource','Vsource_EXT','Vdrain','Vdrain_EXT','Vgate','Vgate_EXT',\
                        'Vbulk','Vbulk_EXT','Isource','Idrain','Igate','Ibulk']
                # temp['file']=pd.Series(temp.index, index=temp.index)
                # temp['file']=name
                df=temp.dropna()
            return df
        else:
            print('alpha.cur file found.')
            with zipfile.ZipFile(filename, 'r') as zf:
                tmp=pd.read_table(zf.open(cur),sep=' ',header=250,skipinitialspace=True,index_col=None,\
                                  usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                           21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                           41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60],
                                  names=['Vsource','Vsource_EXT','Vdrain','Vdrain_EXT','Vgate','Vgate_EXT',\
                              'Vbulk','Vbulk_EXT','Isource','Idrain','Igate','Isub',\
                              'InTsource','IpTsource','IdTsource','IcTsource','ITsource','InTdrain','IpTdrain','IdTdrain',\
                             'IcTdrain','ITdrain','InTgate','IpTgate','IdTgate','IcTgate','ITgate','InTsub','IpTsub','IdTsub',\
                             'IcTsub','ITsub','Qn_sub_SUB(P)','Qp_sub_SUB(P)','QTotal_sub_SUB(P)','Qn_source_SOI(N)','Qp_source_SOI(N)',\
                             'QTotal_source_SOI(N)','Qn_SOI(P)','Qp_SOI(P)','QTotal_SOI(P)','Qn_drain_SOI(N)','Qp_drain_SOI(N)','QTotal_drain_SOI(N)',\
                             'Qn_upper_source(N)','Qp_upper_source(N)','QTotal_upper_source(N)','Qn_upper_drain(N)','Qp_upper_drain(N)',\
                             'QTotal_upper_drain(N)','Qn_upperupper_source(N)','Qp_upperupper_source(N)','QTotal_upperupper_source(N)',\
                             'Qn_gate_GatePoly(M)','Qp_gate_GatePoly(M)','QTotal_gate_GatePoly(M)','Qn_upperupper_drain(N)',\
                              'Qp_upperupper_drain(N)','QTotal_upperupper_drain(N)','Func_Alpha1'])
                time=pd.read_table(zf.open(cur),sep=' ',header=250,skipinitialspace=False,index_col=None,\
                                  usecols=[1],names=['t'])
                df=tmp.dropna()
                df=pd.concat([df,time.dropna()],axis=1)
            return df
    else:
        pass

def plot_IdVg(df,label=''):
    fig=plt.figure(facecolor ="#FFFFFF",figsize=(4, 3))
    plt.axes(facecolor ="#FFFFFF")
    plt.style.use('classic')

    plt.plot(df['Vgate'],df['Idrain'],'r',label=label)
    plt.yscale('log')
    plt.ylabel('Idrain [A/um]')
    plt.xlabel('Vdrain [V]')
    plt.legend(loc=2,prop={'size':6})
    plt.xlim([0,1.2])
    plt.show()

def plot_alpha(df,label=''):
    fig=plt.figure(facecolor ="#FFFFFF",figsize=(4, 3))
    plt.axes(facecolor ="#FFFFFF")
    plt.style.use('classic')

    plt.plot(df['t'],df['Idrain'],'b',label=label)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Idrain [A/um]')
    plt.xlabel('time [s]')
    plt.legend(loc=2,prop={'size':6})
    # plt.xlim([0,1.2])
    plt.show()

def read_SRIM(thick=0.2,filename1="",filename2=""):
    #すべてのSRIM outputには対応していない
    if filename1=="":
        filename1=easygui.fileopenbox(msg='Open ion in Gold file', title=None, default='*', multiple=False)
        filename2=easygui.fileopenbox(msg='Open ion in Silicon file', title=None, default='*', multiple=False)
    else:
        pass
    print filename1.encode('utf-8')
    print filename2.encode('utf-8')

    # filename1
    col_names = ['Energy','Energy_unit','de/dx_elec','de/dx_nuc','Range','Range_unit','Longitudinal_straggling','Longitudinal_straggling_unit','Lateral_straggling','Lateral_straggling_unit','A']
    df1=pd.read_csv(filename1,sep=' ',header='infer',skiprows=24,skipinitialspace=True,skipfooter=13,names=col_names)
    del df1['A']
    df1['Energy'][df1['Energy_unit']=='keV']=df1['Energy'][df1['Energy_unit']=='keV'].copy()/1000
    # df['Energy_unit'][df['Energy_unit']=='keV']='MeV'
    df1['Range'][df1['Range_unit']=='A']=df1['Range'][df1['Range_unit']=='A'].copy()/10000
    # df['Range_unit'][df['Range_unit']=='A']='um'
    df1['Range_invert']= -df1['Range']+df1['Range'].iloc[-1]

    # filename2
    col_names = ['Energy','Energy_unit','de/dx_elec','de/dx_nuc','Range','Range_unit','Longitudinal_straggling','Longitudinal_straggling_unit','Lateral_straggling','Lateral_straggling_unit','A']
    df2=pd.read_csv(filename2,sep=' ',header='infer',skiprows=24,skipinitialspace=True,skipfooter=13,names=col_names)
    del df2['A']
    df2['Energy'][df2['Energy_unit']=='keV']=df2['Energy'][df2['Energy_unit']=='keV'].copy()/1000
    # df['Energy_unit'][df['Energy_unit']=='keV']='MeV'
    df2['Range'][df2['Range_unit']=='A']=df2['Range'][df2['Range_unit']=='A'].copy()/10000
    # df['Range_unit'][df['Range_unit']=='A']='um'
    df2['Range_invert']= -df2['Range']+df2['Range'].iloc[-1]

    # thick #Gold fiol
    f = interpolate.interp1d(df1['Range_invert'], df1['Energy'])
    energy=f(thick)
    print(str(energy)+' [MeV] after ' +str(thick)+ ' um Au foil')

    fig=plt.figure(facecolor ="#FFFFFF",figsize=(16, 10))
    plt.style.use('classic')
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()

    ax1.plot(df1['Range_invert'],df1['de/dx_elec'])
    ax2.plot(df1['Range_invert'],df1['Energy'],'r')
    ax1.set_ylabel('LET [MeV/(mg/cm2)]')
    ax1.yaxis.label.set_color('blue')
    ax2.set_ylabel('Energy [MeV]')
    ax2.yaxis.label.set_color('red')
    ax1.set_xlabel('Range [um]')
    ax1.grid()
    ax1.axvline(thick)
    # ax1.set_ylim([0,6])
    # ax2.set_ylim([0,30])
    # ax1.set_xlim([0,35])
    # plt.title(os.path.basename(filename1))
    plt.grid()

    g = interpolate.interp1d(df2['Energy'],df2['Range_invert'])
    h = interpolate.interp1d(df2['Range_invert'], df2['de/dx_elec'])
    df3=pd.DataFrame([[float(energy),g(float(energy)), h(g(float(energy)))]], columns=['Energy','Range_invert','de/dx_elec'])
    df4=df2[df2['Energy']<float(energy)].append(df3,ignore_index=True)

    ax3 = fig.add_subplot(212)
    ax4 = ax3.twinx()
    ax3.plot(df2['Range_invert'],df2['de/dx_elec'])
    ax4.plot(df2['Range_invert'],df2['Energy'],'r')
    ax3.set_ylabel('LET [MeV/(mg/cm2)]')
    ax3.yaxis.label.set_color('blue')
    ax4.set_ylabel('Energy [MeV]')
    ax4.yaxis.label.set_color('red')
    ax3.set_xlabel('Range [um]')
    ax3.grid()
    # ax1.set_ylim([0,6])
    # ax2.set_ylim([0,30])
    # ax1.set_xlim([0,35])
    # plt.title(os.path.basename(filename2))
    plt.grid()
    plt.show()

    print('range: '+str(df4['Range_invert'].head(1).values-df4['Range_invert'].tail(1).values))
    print('LET@surface: '+str(df4['de/dx_elec'].tail(1).values))
    print('Energy@surface: '+str(float(energy)))
    print('Peak_LET: '+str(df4['de/dx_elec'].max()))
    print('Depth@Peak_LET: '+str(df4['Range_invert'][df4['de/dx_elec'].idxmax()]))

    return [df1,df2,df3,df4]

def SRIM(filename=[]):
    if filename==[]:
        filename=easygui.fileopenbox()
    else:
        print filename.encode('utf-8')
    #すべてのSRIM outputには対応していない
    col_names = ['Energy','Energy_unit','de/dx_elec','dE/dx_nuc','Range','Range_unit','Longitudinal_straggling','Longitudinal_straggling_unit','Lateral_straggling','Lateral_straggling_unit','A']
    df=pd.read_csv(filename,sep=' ',header='infer',skiprows=24,skipinitialspace=True,skipfooter=13,names=col_names)
    del df['A']
    df['Energy'][df['Energy_unit']=='keV']=df['Energy'][df['Energy_unit']=='keV'].copy()/1000
    df['Energy'][df['Energy_unit']=='GeV']=df['Energy'][df['Energy_unit']=='GeV'].copy()*1000
    # df['Energy_unit'][df['Energy_unit']=='keV']='MeV'
    df['Range'][df['Range_unit']=='A']=df['Range'][df['Range_unit']=='A'].copy()/10000
    df['Range'][df['Range_unit']=='mm']=df['Range'][df['Range_unit']=='mm'].copy()*1000
    # df['Range_unit'][df['Range_unit']=='A']='um'
    df['Range_invert']= -df['Range']+df['Range'].iloc[-1]

    fig=plt.figure(facecolor ="#FFFFFF",figsize=(16, 5))
    plt.style.use('classic')
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(df['Range_invert'],df['de/dx_elec'])
    ax2.plot(df['Range_invert'],df['Energy'],'r')
    ax1.set_ylabel('LET [MeV/(mg/cm2)]')
    ax1.yaxis.label.set_color('blue')
    ax2.set_ylabel('Energy [MeV]')
    ax2.yaxis.label.set_color('red')
    ax1.set_xlabel('Range [um]')
    ax1.grid()
    plt.title(os.path.basename(filename))
    plt.grid()
    plt.show()

    print('range: '+str(df['Range_invert'].head(1).values-df['Range_invert'].tail(1).values))
    print('LET@surface: '+str(df['de/dx_elec'].tail(1).values))
    print('Energy@surface: '+str(df['Energy'].tail(1).values))
    print('Peak_LET: '+str(df['de/dx_elec'].max()))
    print('Depth@Peak_LET: '+str(df['Range_invert'][df['de/dx_elec'].idxmax()]))
    # print('Energy@surface: '+str(float(energy)))

    #dataframe for result
    result=pd.DataFrame(
                        {'filename': [filename],
                        'range': [df['Range_invert'].head(1).values-df['Range_invert'].tail(1).values],
                         'LET@surfac': [df['de/dx_elec'].tail(1).values],
                         'Energy@surface': [df['Energy'].tail(1).values],
                         'Peak_LET': [df['de/dx_elec'].max()],
                         'Depth@Peak_LET': [df['Range_invert'][df['de/dx_elec'].idxmax()]]
                         })

    return df,result

def plot_waveform(filename=[],option=False):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass
    if filename:
        if option:
            print filename.encode('utf-8')
        else:
            pass
        names=['t','XOR','CLK','SMUX_IN','SMUX_OUT8','SMUX_OUT7','SMUX_OUT6','SMUX_OUT5','SMUX_OUT4','SMUX_OUT3','SMUX_OUT2','SMUX_OUT1']
        df=pd.read_csv(filename,sep='\t',skiprows=22,names=names)
        if len(df.index) == 0:
            if option:
                print u'blank file!'
            else:
                pass
        else:
            dt0=8.000000E-10 #delta t for XOR
            dt1=2.000000E-9  #delta t for degital signal
            df.t=df.index*dt0*10**6  #us
            df.t1=df.index*dt1*10**6 #us
            fig=plt.figure(facecolor ="#FFFFFF",figsize=(16,16))
            plt.style.use('classic')
            f=np.diff(df.CLK)
            CLK_index = np.where(f>0.5)

            def plot_CLK_edge():
                for t in df.t1[CLK_index]:
                    plt.axvline(x=t,linestyle='--')

            xlim=[1,8]

            ax=plt.subplot(11,1,1)
            plt.plot(df.t,df.XOR)
            plt.xticks([])
            plt.yticks([0,5])
            plt.ylim([-1,5])
            plt.xlim(xlim)
            plt.ylabel('XOR')
            plt.title(os.path.basename(filename))

            plt.subplot(11,1,2)
            plt.plot(df.t1,df.CLK)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.xlim(xlim)
            plt.ylabel('CLK')

            plt.subplot(11,1,3)
            plt.plot(df.t1,df.SMUX_IN)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_IN')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,4)
            plt.plot(df.t1,df.SMUX_OUT8)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT8')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,5)
            plt.plot(df.t1,df.SMUX_OUT7)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT7')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,6)
            plt.plot(df.t1,df.SMUX_OUT6)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT6')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,7)
            plt.plot(df.t1,df.SMUX_OUT5)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT5')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,8)
            plt.plot(df.t1,df.SMUX_OUT4)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT4')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,9)
            plt.plot(df.t1,df.SMUX_OUT3)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT3')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,10)
            plt.plot(df.t1,df.SMUX_OUT2)
            plt.xticks([])
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT2')
            plot_CLK_edge()
            plt.xlim(xlim)

            plt.subplot(11,1,11)
            plt.plot(df.t1,df.SMUX_OUT1)
            plt.yticks([0,1])
            plt.ylim([-0.1,1.1])
            plt.ylabel('SMUX_OUT1')
            plt.xlabel('time [us]')
            plt.tight_layout()
            plot_CLK_edge()
            plt.xlim(xlim)
            # plt.show()
            fig.savefig(filename[0:-4]+'.png',format='png',dpi=300)
            if option:
                print('Saving... .png format.')
            else:
                pass
            plt.close(fig)
            plt.clf()

        return [df,filename]
    else:
        print('Canceled!')
        df=[]
        return [df,filename]

def plot_waveform_auto():
    filenames=easygui.fileopenbox(msg='Select waveform data.', title=None, default='*', filetypes=['*.csv'], multiple=True)
    for filename in tqdm_notebook(filenames):
        plot_waveform(filename=filename)
    print 'finish!'

def plot_MCA(filename=[],ROI=True):
    if not filename:
        filename=easygui.fileopenbox(msg='Choose .JAC', title=None, default=None,multiple=False)
    else:
        pass

    if filename:
        print filename.encode('utf-8')

        #dataframe for result
        result=pd.DataFrame(index=[],columns=['filename','ROI','Noise','peak'])

        #MCA data plot
        df = pd.read_csv(filename,sep=',',names=['count'])
        df=df.drop([0,1,2])
        df.index=range(4, 1025)
        df[['count']]=df[['count']].astype(int)

        fig=plt.figure(facecolor ="#FFFFFF",figsize=(16, 5))
        plt.style.use('classic')
        plt.bar(df.index,df['count'], width=1.0,edgecolor='none')
        plt.yscale('log')
        plt.xlabel('channel')
        plt.ylabel('Counts')
        plt.xlim([0,1000])
        plt.grid(True)
        plt.title(os.path.basename(filename))
        if ROI:
            x_min=2.0*df['count'][df['count']>1].argmax()-df['count'][df['count']>1].index.max()
            x_max=df['count'][df['count']>1].index.max()
            plt.fill_between(x=[x_min,x_max],y1=1,y2=10000,facecolor='blue', alpha=0.1)
            ROI=df['count'][(x_min < df.index) & (df.index < x_max)].sum()
            noise=df['count'][(x_min > df.index) | (df.index > x_max)].sum()
            print 'ROI      : '+str(ROI)+' counts'
            print 'noise    : '+str(noise)+' counts'
            print 'ratio    : '+str(float(noise)/(ROI+noise)*100)+' %'
            print 'peak ch  : ' +str(df['count'][df['count']>1].argmax())
            tmp=pd.DataFrame(
                    {'filename': [filename],
                    'ROI': [ROI],
                     'Noise': [noise],
                     'peak': [df['count'][df['count']>1].argmax()]
                     })
            result=pd.concat([result, tmp], ignore_index=True)
        else:
            pass
        # plt.title(filename)
        plt.show()
    else:
        result=[]
        df=[]

    return df,result

def analyse_waveform(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass
    if filename:
        print filename.encode('utf-8')
        names=['t','XOR','CLK','SMUX_IN','SMUX_OUT8','SMUX_OUT7','SMUX_OUT6','SMUX_OUT5','SMUX_OUT4','SMUX_OUT3','SMUX_OUT2','SMUX_OUT1']
        df=pd.read_csv(filename,sep='\t',skiprows=22,names=names)
        if len(df.index) == 0:
            print u'blank file!'
            error_name=np.nan
            duration=np.nan
            error_mode=np.nan
        else:
            dt0=8.000000E-10 #delta t for XOR
            dt1=2.000000E-9  #delta t for degital signal
            df.t=df.index*dt0*10**6  #us
            df.t1=df.index*dt1*10**6 #us
            if len(df[['SMUX_IN','SMUX_OUT1','SMUX_OUT2','SMUX_OUT3','SMUX_OUT4','SMUX_OUT5','SMUX_OUT6','SMUX_OUT7','SMUX_OUT8']].dropna().corr().dropna(axis=0,how='all').index)==1:
                error_name=df[['SMUX_IN','SMUX_OUT1','SMUX_OUT2','SMUX_OUT3','SMUX_OUT4','SMUX_OUT5','SMUX_OUT6','SMUX_OUT7','SMUX_OUT8']].dropna().corr().dropna(axis=0,how='all').index.values[0]
                if df.SMUX_IN.mean()<0.5:
                    error_mode='LowToHigh'
                else:
                    error_mode='HighToLow'
            else:
                error_name='other'
                error_mode='other'
            # print error_name
            XOR=np.logical_xor((df.SMUX_OUT1.dropna()==1),(df.SMUX_OUT2.dropna()==1))\
                | np.logical_xor((df.SMUX_OUT3.dropna()==1),(df.SMUX_OUT4.dropna()==1))\
                | np.logical_xor((df.SMUX_OUT5.dropna()==1),(df.SMUX_OUT6.dropna()==1))\
                | np.logical_xor((df.SMUX_OUT7.dropna()==1),(df.SMUX_OUT8.dropna()==1))
            duration=dt1*len(XOR[XOR==True])
            # print str(duration)+'s'
        return [error_name,duration,error_mode]

def analyse_waveform_auto():
    filenames=easygui.fileopenbox(msg='Select waveform data.', title=None, default='*', filetypes=['*.csv'], multiple=True)
    names=['filename','error','duration','test','error_mode']
    df=pd.DataFrame(index=[],columns=names)
    for filename in filenames:
        error_name,duration,error_mode=analyse_waveform(filename=filename)
        tmp=pd.DataFrame(
                {'filename': [os.path.basename(filename)],
                'error': [error_name],
                 'duration': [duration],
                 'test': [os.path.dirname(filename)[-5:]],
                 'error_mode': [error_mode]
                 })
        df=pd.concat([df, tmp], ignore_index=True)
    print df
    # print df.describe()
    # print df.error.value_counts()
    # print df.error_mode.value_counts()
    # print os.path.dirname(filename)[-5:]
    df.to_csv('C:/Users/14026/Desktop/'+os.path.dirname(filename)[-5:]+'.csv')
    print 'saved at'+'C:/Users/14026/Desktop/'+os.path.dirname(filename)[-5:]+'.csv'
    print 'finish!'
    print df.pivot_table(values = 'duration',
               index = ['error_mode'], columns = ['error'],
               aggfunc = 'count',fill_value = 0)
    return df

def add_SET_data(filename=[]):
    print('Starting... add_SET_data.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=False)
    else:
        pass

    if filename:
        os.chdir(os.path.dirname(filename))
        os.chdir('..')
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        ion=data.irradiation['ion'].values
        print(ion+formatted+'.mydata is seleced')
        filenames_SET=easygui.fileopenbox(msg='Open '+ion+formatted+' SET data.', title=None, default='*', filetypes=['*.csv'], multiple=False)
        if filenames_SET:
            data.SET=pd.DataFrame(index=[], columns=['DateTime'])
            if 'SET' in data.field:
                pass
            else:
                data.field.append('SET')
            names=['id','duration','error','error_mode','filename','test']
            df=pd.DataFrame(index=[],columns=names)
            df=pd.read_csv(filenames_SET,names=names,skiprows=1)
            data.SET=df
            data.SET['time']=data.SET.filename.map(lambda x: pd.to_datetime(x[0:14]))
            print data.SET
            #save SEE data
            save_data(data,filename)
            print('Saving... SET data at '+filename.encode('utf-8'))

        else:
            print('Canceled.')
    else:
        pass
    print('End of add_SET_data.')

def read_goes_Xray(filenames=[]):
    print('Starting... read_goes_Xray.')
    if not(filenames):
        filenames=easygui.fileopenbox(msg='Open GOES Xray  .cdf files.', title=None, default='*', filetypes=['*.cdf'], multiple=True)
    else:
        pass
    data=pd.DataFrame(index=[], columns=['time','A_AVG','B_AVG'])
    if filenames:
        for filename in filenames:
            print filename.encode('utf-8')
            nc = netCDF4.Dataset(filename, 'r')
            tmp=pd.DataFrame({ 'time':nc.variables['time_tag'][:],
                           'A_AVG' :nc.variables['A_AVG'][:],
                          'B_AVG' :nc.variables['B_AVG'][:]
                              })
            nc.close()
            data=pd.concat([data, tmp], ignore_index=True)
            label='XRS long wavelength channel irradiance (0.1-0.8 nm)'
            unit=r'$W/m^2$'
    else:
        print('Canceled.')
    return [data,label,unit]

def read_goes_proton(filenames=[]):
    print('Starting... read_goes_proton.')
    if not(filenames):
        filenames=easygui.fileopenbox(msg='Open GOES proton .cdf files.', title=None, default='*', filetypes=['*.cdf'], multiple=True)
    else:
        pass
    data=pd.DataFrame(index=[], columns=['time','P1W','P2W','P3W','P4W','P5W','P6W','P7W'])
    if filenames:
        for filename in filenames:
            print filename.encode('utf-8')
            nc = netCDF4.Dataset(filename, 'r')
            tmp=pd.DataFrame({ 'time':nc.variables['time_tag'][:],
                       'P1W' :nc.variables['P1W_UNCOR_FLUX'][:],
                      'P2W' :nc.variables['P2W_UNCOR_FLUX'][:]
#                       'P3W' :nc.variables['P3W_UNCOR_FLUX'][:],
#                       'P4W' :nc.variables['P4W_UNCOR_FLUX'][:],
#                       'P5W' :nc.variables['P4W_UNCOR_FLUX'][:],
#                       'P6W' :nc.variables['P6W_UNCOR_FLUX'][:],
#                       'P7W' :nc.variables['P7W_UNCOR_FLUX'][:]
                     })
            nc.close()
            data=pd.concat([data, tmp], ignore_index=True)
            label=['p1A(2.5 MeV)','p2A(6.5 MeV)','p3A(11.6 MeV)','p4A(30.6 MeV)','p5A(63.1 MeV)','p6A(165 MeV)','p7A(433 MeV)']
            unit = r'$p/(cm^2 s sr MeV)$'
    else:
        print('Canceled.')
    return [data,label,unit]

def read_SCi_data(filenames=[]):
    print('Starting... read_SCi_data.')
    if not(filenames):
        filenames=easygui.fileopenbox(msg='Open SCi .tar files.', title=None, default='*', filetypes=['*.tar'], multiple=True)
    else:
        pass
    #空のDataframe作成
    error = pd.DataFrame([], columns=['ALOS_2','SF_BSRAM_1BIT_ERR_CNT','SF_BSRAM_2BIT_ERR_CNT','SF_D-CACHE_ERR_CNT','SF_IN-CACHE_ERR_CNT','SF_OTHER_ERR_CNT'])
    if filenames:
        for file in filenames:
            print file.encode('utf-8')
            ## tar/tar.gz/tar.bz2ファイルを読む
            tf = tarfile.open(file, 'r')
            ## ファイル情報(tarinfo)をすべて取り出す
            for ti in tf:
                print ti.name
                t = tf.getmember(ti.name)
                tmp=pd.read_table(tf.extractfile(t)\
                        ,sep=',',skiprows=4,index_col=[0],\
                        names=['time','ALOS_2','SF_BSRAM_1BIT_ERR_CNT','SF_BSRAM_2BIT_ERR_CNT','SF_D-CACHE_ERR_CNT','SF_IN-CACHE_ERR_CNT','SF_OTHER_ERR_CNT'])
            error=pd.concat([error,tmp])

    else:
        print('Canceled.')
    error.index=pd.to_datetime(error.index)
    error.index.name='time'
    return error

def count_SET(filename=[]):
    print('Starting... count_SET.')
    if not(filename):
        filename=easygui.fileopenbox(msg='Open bmp file.', title=None, default='*', filetypes=['*.bmp'], multiple=False)
    else:
        pass

    if filename:
        print filename.encode('utf-8')
        im = Image.open(filename)
        gray=np.array(im.crop((353,0,905,552)).convert('L'))
        num=np.round(len(gray[gray>128])/5.52/5.52)
        print num

    else:
        pass

    return [num,gray]

def read_IV(filename=[],type='IdVg',show_Vt=False,absolute=True,scale='log',ylim=[1e-12,1e-2],png=True,eps=False,show=True,option=True,comment='',device='NMOS',length=100,place='0,0',width=100):
    L=length
    W=width
    if option:
        print('Starting... read_IdVg.')
    else:
        pass
    if not(filename):
        filename=easygui.fileopenbox(msg='Open .txt file.', title=None, default='*', filetypes=['*.txt'], multiple=False)
    else:
        pass
    if filename:
        if option:
            print filename.encode('utf-8')
        else:
            pass
        df=pd.read_csv(filename, sep='\t',skiprows=15)
        header=pd.read_csv(filename, sep='\t',nrows=7,skiprows=4,names=['item','value'])
        name=[]
        for i in ['Is','Ig','Id','Isub']:
            for j in np.arange(1,(len(df.columns)-1)/4+1,1):
        #         print i+'_'+str(j)
                # print (header.value[5]-header.value[4])/(header.value[6]-1)*j
                if type=='IdVg':
                    name.append(i+'@Vd='+str(header.value[4]+header.value[5]*(j-1))+'V')
                else:
                    name.append(i+'@Vg='+str(header.value[4]+header.value[5]*(j-1))+'V')
        name.append('dummy')
        df=pd.read_csv(filename, sep='\t',skiprows=16,names=name)
        del df['dummy']
        df.index.name='V'
        if show_Vt&(type=='IdVg'):
            try:
                # select Is to calc Vt
                if device=='NMOS':
                    x1=np.abs(df.ix[(np.abs(df.ix[:,0])>=W/L*1e-6),0]).head(1).index.values[0]
                    x2=np.abs(df.ix[np.abs(df.ix[:,0])<W/L*1e-6,0]).tail(1).index.values[0]
                    y1=np.abs(df.ix[(np.abs(df.ix[:,0])>=W/L*1e-6),0]).head(1).values[0]
                    y2=np.abs(df.ix[np.abs(df.ix[:,0])<W/L*1e-6,0]).tail(1).values[0]
                    Vt=(x2-x1)/(y2-y1)*(W/L*1e-6-y2)+x2
                else:
                    x1=df.ix[(df.ix[:,0]>=W/L*1e-6),0].head(1).index.values[0]
                    x2=df.ix[df.ix[:,0]<W/L*1e-6,0].tail(1).index.values[0]
                    y1=df.ix[(df.ix[:,0]>=W/L*1e-6),0].head(1).values[0]
                    y2=df.ix[df.ix[:,0]<W/L*1e-6,0].tail(1).values[0]
                    Vt=(x2-x1)/(y2-y1)*(W/L*1e-6-y2)+x2
                if option:
                    print 'Vt='+str(Vt)+ ' [V]'
                else:
                    pass
            except:
                Vt=0
                if option:
                    print 'Can not calculate Vt.'
                else:
                    pass
        else:
            Vt=0
            if option:
                print 'Vt was not calculated.'
            else:
                pass
        n=(len(df.columns))/4
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(10,10))
        plt.style.use('classic')
        #leftTop
        plt.subplot(2,2,1)
        for i in np.arange(0,n,1):
            if absolute:
                plt.plot(df.index,abs(df.ix[:,i]),'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
            else:
                plt.plot(df.index,df.ix[:,i],'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
        plt.yscale(scale)
        plt.ylim(ylim)
        # plt.xlim([0,10])
        # plt.xlabel(r'$Vg [V]$')
        plt.ylabel(r'$Is [A]$')
        if type=='IdVg':
            # plt.title('Is-Vg')
            plt.xlabel(r'$Vg [V]$')
            plt.legend(loc=3,fontsize=10)
        else:
            # plt.title('Is-Vd')
            plt.xlabel(r'$Vd [V]$')
            plt.legend(loc=1,fontsize=10)
        if show_Vt:
            plt.axhline(W/L*1e-6, linestyle='dashed', linewidth=1)
            plt.axvline(Vt, linestyle='dashed', linewidth=1)
        else:
            pass
        plt.grid()
        #RightTop
        plt.subplot(2,2,2)
        for i in np.arange(0+n,n+n,1):
            if absolute:
                plt.plot(df.index,abs(df.ix[:,i]),'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
            else:
                plt.plot(df.index,df.ix[:,i],'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
        plt.yscale(scale)
        plt.ylim(ylim)
        # plt.xlim([0,10])
        plt.xlabel(r'$Vg [V]$')
        plt.ylabel(r'$Ig [A]$')
        if type=='IdVg':
            # plt.title('Ig-Vg')
            plt.xlabel(r'$Vg [V]$')
            plt.legend(loc=3,fontsize=10)
        else:
            # plt.title('Id-Vd')
            plt.xlabel(r'$Vd [V]$')
            plt.legend(loc=1,fontsize=10)
        plt.grid()
        #LeftBottom
        plt.subplot(2,2,3)
        for i in np.arange(0+2*n,n+2*n,1):
            if absolute:
                plt.plot(df.index,abs(df.ix[:,i]),'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
            else:
                plt.plot(df.index,df.ix[:,i],'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
        plt.yscale(scale)
        plt.ylim(ylim)
        # plt.xlim([0,10])
        plt.xlabel(r'$Vg [V]$')
        plt.ylabel(r'$Id [A]$')
        if type=='IdVg':
            # plt.title('Id-Vg')
            plt.xlabel(r'$Vg [V]$')
            plt.legend(loc=3,fontsize=10)
        else:
            # plt.title('Id-Vd')
            plt.xlabel(r'$Vd [V]$')
            plt.legend(loc=1,fontsize=10)
        if show_Vt&(type=='IdVg'):
            plt.axhline(W/L*1e-6, linestyle='dashed', linewidth=1)
            plt.axvline(Vt, linestyle='dashed', linewidth=1)
        else:
            pass
        plt.grid()
        #RightBottom
        plt.subplot(2,2,4)
        for i in np.arange(0+3*n,n+3*n,1):
            if absolute:
                plt.plot(df.index,abs(df.ix[:,i]),'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
            else:
                plt.plot(df.index,df.ix[:,i],'o-',markersize=5, markeredgecolor=None,alpha=0.5,markeredgewidth=0)
        plt.yscale(scale)
        plt.ylim(ylim)
        # plt.xlim([0,10])
        plt.xlabel(r'$Vg [V]$')
        plt.ylabel(r'$Isub [A]$')
        if type=='IdVg':
            # plt.title('Id-Vg')
            plt.xlabel(r'$Vg [V]$')
            plt.legend(loc=3,fontsize=10)
        else:
            # plt.title('Id-Vd')
            plt.xlabel(r'$Vd [V]$')
            plt.legend(loc=1,fontsize=10)
        plt.grid()
        plt.suptitle(comment+'_L='+str(length),fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        if show:
            plt.show()
        else:
            pass
        #save figure as png
        if png:
            fig.savefig(filename[0:-4]+'.png',format='png',dpi=300)
            if option:
                print('Saving... .png format.')
            else:
                pass
        else:
            pass
        #save as eps
        if eps:
            fig.savefig(filename[0:-4]+'.eps',format='eps',dpi=300)
            if option:
                print('Saving... .eps format.')
            else:
                pass
        else:
            pass
        plt.close(fig)
        plt.clf()

        # df modify
        df.columns=['Is','Ig','Id','Isub']
        df['condition']=header.value[4]
        df['device']=device
        df['filename']=filename
        df['length']=length
        df['type']=type
        df['Vt']=Vt
        df['V']=df.index
        df['place']=place
    else:
        pass
    return [df,header,Vt]

def read_IV_minimal(filename=[],type='IdVg',option=True):
    if option:
        print('Starting... read_IV_minimal.')
    else:
        pass
    if not(filename):
        filename=easygui.fileopenbox(msg='Open .csv file.', title=None, default='*', filetypes=['*.csv'], multiple=False)
    else:
        pass
    if filename:
        if option:
            print filename.encode('utf-8')
        else:
            pass
        if type=='IdVg':
            df=pd.read_csv(filename, skiprows=229,names=['DataName','Vg','Vd','Id','absId'])
            del df['DataName']
        elif type=='IdVgwIg':
            df=pd.read_csv(filename, skiprows=229,names=['Vg','Vd','Ib','Id','Ig','Is','absId'],usecols=[1,2,3,4,5,6,7])
        else:
            df=pd.read_csv(filename, skiprows=229,names=['DataName','Vd','Vg','Id'])
            del df['DataName']
    else:
        pass

    return df

def plot_IV_minimal(df,type='IdVg',device='NMOS',show_fig=False,save_fig=True,place='',process='',wafer='',length=10,filename=[]):
    fig=plt.figure(facecolor ="#FFFFFF",figsize=(5,5))
    plt.style.use('classic')
    if type=='IdVg':
        for condition in df.Vd.unique():
            plt.plot(df.Vg[df.Vd==condition],np.abs(df.Id)[df.Vd==condition],label='Vd='+str(condition)+' [V]')
        plt.grid()
        plt.yscale('log')
        plt.ylabel('Id [A]')
        plt.xlabel('Vg [V]')
        if device=='NMOS':
            plt.legend(loc=4,fontsize=10)
        else:
            plt.legend(loc=1,fontsize=10)
    elif type=='IdVgwIg':
        plt.rcParams["font.size"] = 8
        plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
        # font = {'family' : 'normal',
        # 'weight' : 'normal',
        # 'size'   : 5}
        #
        # matplotlib.rc('font', **font)
        ax_Id=plt.subplot2grid((2,2),(0,0))
        ax_Ig=plt.subplot2grid((2,2),(0,1))
        ax_Is=plt.subplot2grid((2,2),(1,0))
        ax_Ib=plt.subplot2grid((2,2),(1,1))

        for Vd in [-1,-0.05]:
            ax_Id.plot(df.Vg[df.Vd==Vd],np.abs(df.Id[df.Vd==Vd]),'o-',label='Vd={} [V]'.format(Vd),alpha=0.5,markersize=3)
        ax_Id.set_yscale('log')
        ax_Id.grid()
        ax_Id.set_ylabel('Id')
        ax_Id.set_xlabel('Vg')
        ax_Id.legend(fontsize=5)
        ax_Id.set_ylim([1e-14,1e-3])
        plt.tight_layout()

        for Vd in [-1,-0.05]:
            ax_Ig.plot(df.Vg[df.Vd==Vd],np.abs(df.Ig[df.Vd==Vd]),'o-',label='Vd={} [V]'.format(Vd),alpha=0.5,markersize=3)
        ax_Ig.set_yscale('log')
        ax_Ig.grid()
        ax_Ig.set_ylabel('Ig')
        ax_Ig.set_xlabel('Vg')
        ax_Ig.set_ylim([1e-14,1e-3])
        ax_Ig.legend(fontsize=5)
        plt.tight_layout()

        for Vd in [-1,-0.05]:
            ax_Is.plot(df.Vg[df.Vd==Vd],np.abs(df.Is[df.Vd==Vd]),'o-',label='Vd={} [V]'.format(Vd),alpha=0.5,markersize=3)
        ax_Is.set_yscale('log')
        ax_Is.grid()
        ax_Is.set_ylabel('Is')
        ax_Is.set_xlabel('Vg')
        ax_Is.set_ylim([1e-14,1e-3])
        ax_Is.legend(fontsize=5)
        plt.tight_layout()

        for Vd in [-1,-0.05]:
            ax_Ib.plot(df.Vg[df.Vd==Vd],np.abs(df.Ib[df.Vd==Vd]),'o-',label='Vd={} [V]'.format(Vd),alpha=0.5,markersize=3)
        ax_Ib.set_yscale('log')
        ax_Ib.grid()
        ax_Ib.set_ylabel('Ib')
        ax_Ib.set_xlabel('Vg')
        ax_Ib.set_ylim([1e-14,1e-3])
        ax_Ib.legend(fontsize=5)
        plt.tight_layout()

        # plt.tight_layout()



    elif type=='IdVd':
        plt.grid()
        plt.ylabel('Id [A]')
        plt.xlabel('Vd [V]')
        if device=='NMOS':
            for condition in np.round(df.Vg.unique()[df.Vg.unique()>=0],decimals=3):
                plt.plot(df.Vd[df.Vg==condition],np.abs(df.Id)[df.Vg==condition],label='Vg='+str(condition)+' [V]')
            plt.legend(loc=2,fontsize=6)
        else:
            for condition in np.round(df.Vg.unique()[df.Vg.unique()<=0],decimals=3):
                plt.plot(df.Vd[df.Vg==condition],np.abs(df.Id)[df.Vg==condition],label='Vg='+str(condition)+' [V]')
            plt.legend(loc=1,fontsize=6)

    else:
        pass
    if type=='IdVgwIg':
        # plt.suptitle('{0}_wafer{1}_{2}_{3}_L={4}um_{5}'.format(process,wafer,place,device,length,type))
        # plt.subplots_adjust(top=0.5)
        plt.tight_layout()
    else:
        plt.title('{0}_wafer{1}_{2}_{3}_L={4}um_{5}'.format(process,wafer,place,device,length,type),fontsize=10)

    # plt.tight_layout()
    if save_fig:
        plt.savefig(filename[0:-4]+'.png',format='png',dpi=300)
    if show_fig:
        plt.show()
    plt.close(fig)
    plt.clf()

def calc_Vt(df,option=False,device='NMOS',show_fig=False):
    """
    fuction of Vt calculation. Besed on Id equation in saturation region.
    lower:peak of grad(sqrt(Id))
    upper:peak of grad(Id)
    """
    if device=='NMOS':
        lower=df.V[np.sqrt(df.Id).diff().argmax()]
        upper=df.V[df.Id.diff().argmax()]
        Gm_max=df.Id.diff().max()
        y=df.Id[(lower<df.V)&(df.V<upper)]
        x=df.V[(lower<df.V)&(df.V<upper)]
        # print lower,upper,x,y
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        Vt=-intercept/slope
    else:
        upper=df.V[np.sqrt(np.abs(df.Id)).diff().argmax()]
        lower=df.V[np.abs(df.Id).diff().argmax()]
        Gm_max=np.abs(df.Id).diff().max()
        y=np.abs(df.Id)[(lower<df.V)&(df.V<upper)]
        x=df.V[(lower<df.V)&(df.V<upper)]
        # print lower,upper,x,y
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        Vt=-intercept/slope
    if option:
        print '****************Vt calculation**********************'
        print 'Reference V region between [{0},{1}]'.format(lower,upper)
        print 'Vt={:.3f} [V]'.format(Vt)
        print 'Gm={:.3e} [S]'.format(Gm_max)
        print '****************Vt calculation**********************'
    if show_fig:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(5,5))
        plt.style.use('classic')
        plt.plot(df.V,np.abs(df.Id),label='input')
        if device=='NMOS':
            t=np.arange(0,5,0.1)
        else:
            t=np.arange(-5,0,0.1)
        plt.plot(t,slope*t+intercept,label='fit')
        plt.grid()
        plt.yscale('linear')
        # plt.xlim([0,2])
        # plt.gca().set_ylim()
        # plt.gca().set_ylim(bottom=0)
        plt.ylabel('Id [A]')
        plt.xlabel('Vg [V]')
        plt.legend(fontsize=10)
        left, bottom, width, height = [0.5, 0.25, 0.3, 0.3]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.grid()
        ax2.tick_params(axis='both', which='major', labelsize=5)
        if device=='NMOS':
            ax2.plot(df.V[(0<df.V)&(df.V<1.5)],df.Id[(0<df.V)&(df.V<1.5)],label='input')
        else:
            ax2.plot(df.V[(-1.5<df.V)&(df.V<0)],np.abs(df.Id)[(-1.5<df.V)&(df.V<0)],label='input')
        t=np.arange(Vt-0.5,Vt+0.5,0.05)
        ax2.plot(t,slope*t+intercept,label='fit')
        ax2.set_ylim(bottom=0)
        ax2.set_xlim([Vt-0.5,Vt+0.5])
        plt.show()
    return [Vt,slope,intercept,Gm_max]

def calc_Vt_lapis(df,option=False,device='NMOS',L=10,W=100,show_fig=False):
    """
    fuction of Vt calculation. Besed on lapis semiconductor method.
    Reference Id is W/L*1e-7 [A].
    """
    ref=1e-7
    if device=='NMOS':
        x1=df.V[np.abs(df.Id)>=W/L*ref].head(1).values[0]
        x2=df.V[np.abs(df.Id)<W/L*ref].tail(1).values[0]
        y1=df.Id[df.V[np.abs(df.Id)>=W/L*ref].head(1).index.values[0]]
        y2=df.Id[df.V[np.abs(df.Id)<W/L*ref].tail(1).index.values[0]]
        Vt=(x2-x1)/(y2-y1)*(W/L*ref-y2)+x2
    else:
        x1=df.V[np.abs(df.Id)>=W/L*ref].head(1).values[0]
        x2=df.V[np.abs(df.Id)<W/L*ref].tail(1).values[0]
        y1=np.abs(df.Id)[df.V[np.abs(df.Id)>=W/L*ref].head(1).index.values[0]]
        y2=np.abs(df.Id)[df.V[np.abs(df.Id)<W/L*ref].tail(1).index.values[0]]
        Vt=(x2-x1)/(y2-y1)*(W/L*ref-y2)+x2
    if option:
        print '****************Vt(lapis) calculation***************'
        print 'x1={0},x2={1},y1={2},y2={3}'.format(x1,x2,y1,y2)
        print 'W={0}\nL={1}'.format(W,L)
        print 'Reference Id was {:.2e} [A]'.format(W/L*ref)
        print 'Vt={:.3f} [V]'.format(Vt)
        print '****************Vt(lapis) calculation***************'
    if show_fig:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(5,5))
        plt.style.use('classic')
        plt.plot(df.V,np.abs(df.Id),label='input')
        plt.axhline(W/L*ref,linestyle='--')
        plt.axvline(Vt,linestyle='--')
        plt.grid()
        plt.yscale('log')
        # plt.xlim([0,Vt])
        plt.ylabel('Id [A]')
        plt.xlabel('Vg [V]')
        plt.legend(fontsize=10)
        plt.show()
    return Vt

def calc_SS(df,Vt=1,option=False,device='NMOS',show_fig=False):
    """
    fuction of S.S.(subthreshold slope in mV/decade) calculation. Besed on Id equation in subthreshold region.
    lower:peak of grad(sqrt(Id))
    upper:peak of grad(Id)
    """
    if device=='NMOS':
        def exponential(x,A,x0,B):
            return A*np.exp(B*(x-x0))
        model=lmfit.Model(exponential)
        params = model.make_params()
        params.add('A', value=df.Id[(0<=df.V)&(df.V<Vt)].mean(), min=0, max=np.inf)
        params.add('x0', value=Vt, min=0, max=np.inf)
        params.add('B', value=10.0, min=0, max=np.inf)
        x=df.V[(0<=df.V)&(df.V<Vt)]
        y=df.Id[(0<=df.V)&(df.V<Vt)]
        # print df.V[(df.V<Vt)],df.Id[(df.V<Vt)]
        # print x,y
        fit = model.fit(y, params, x=x,fit_kws={'nan_policy':'omit'})
        SS=1.0/fit.best_values['B']*1000*np.log(10)
    else:
        def exponential(x,A,x0,B):
            return A*np.exp(B*(x-x0))
        model=lmfit.Model(exponential)
        params = model.make_params()
        params.add('A', value=np.mean(np.abs(df.Id)[(Vt/2<df.V)&(df.V<0)]), min=0, max=np.inf)
        params.add('x0', value=Vt, min=-np.inf, max=0)
        params.add('B', value=-10.0, min=-np.inf, max=0)
        x=df.V[(Vt/2<df.V)&(df.V<0)]
        y=np.abs(df.Id)[(Vt/2<df.V)&(df.V<0)]
        # print x,y
        fit = model.fit(y, params, x=x,fit_kws={'nan_policy':'omit'})
        SS=1.0/fit.best_values['B']*1000*np.log(10)
    if option:
        print '****************SS calculation**********************'
        print fit.fit_report()
        print 'Subthreshold slope={:.3f} [mV/decade]'.format(SS)
        print '****************SS calculation**********************'
    if show_fig:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(5,5))
        plt.style.use('classic')
        plt.plot(df.V,np.abs(df.Id),'o-',alpha=0.5,label='input')
        if device=='NMOS':
            t=np.arange(0.5,2,0.1)
        else:
            t=np.arange(Vt/2,0,0.01)
        plt.plot(t,exponential(t,**fit.values),'-',label='fit')
        plt.grid()
        plt.yscale('log')
        # plt.xlim([Vt,0])
        plt.ylabel('Id [A]')
        plt.xlabel('Vg [V]')
        plt.legend(fontsize=10)
        plt.show()
    return [SS,fit]

def read_CV(filename=[],show=True,png=True,eps=False,option=True):
    if option:
        print('Starting... read_CV.')
    else:
        pass
    if not(filename):
        filename=easygui.fileopenbox(msg='Open .csv file.', title=None, default='*', filetypes=['*.csv'], multiple=False)
    else:
        pass
    df=[]
    if filename:
        if option:
            print filename.encode('utf-8')
        else:
            pass
        df=pd.read_csv(filename, sep='\s+',names=['V', 'C'],skiprows=2,skip_footer=1,engine='python')
        Cmax=max(df.C)
        Cmin=min(df.C)
        index=np.argmax((df.C/df.C.tail(1).values[0]).rolling(window=3, min_periods=3).mean().diff())
        x = df.V[index-3:index+3]
        y = (df.C/df.C.tail(1).values[0])[index-3:index+3]
        slope,intercept,rvalue,pvalue,stderr=linregress(x,y) #x and y are arrays or lists.
        Vth=(Cmin/Cmax-intercept)/slope
        param=pd.DataFrame({ 'Cmin' : [Cmin],
                        'Cmax' : [Cmax],
                        'Vth' : [Vth]})

        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        plt.style.use('classic')
        plt.plot(df.V,df.C/df.C.tail(1).values[0],'o')
        plt.plot(df.V,slope*df.V+intercept,'g')
        plt.axvline(Vth)
        # plt.yscale('log')
        plt.ylim([0,1.2])
        plt.xlabel(r'$Voltage [V]$')
        plt.ylabel(r'$C/C_{ox}$')
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()
        else:
            pass
        #save figure as png
        if png:
            fig.savefig(filename[0:-4]+'.png',format='png',dpi=300)
            if option:
                print('Saving... .png format.')
            else:
                pass
        else:
            pass
        #save as eps
        if eps:
            fig.savefig(filename[0:-4]+'.eps',format='eps',dpi=300)
            if option:
                print('Saving... .eps format.')
            else:
                pass
        else:
            pass
        plt.close(fig)
        plt.clf()

    else:
        pass

    return [df,param]


def plot_SET_duration_by_channel(filename=[],type='HighToLow',yscale='log',ylim=[1e-9,1e-6]):
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        df= pd.DataFrame(index=[], columns=[])
        for file in filenames:
            tmp=load_data(filename=file)
            # print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            df=pd.concat([df,tmp.SET],ignore_index=True)
    else:
        pass
    fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
    plt.style.use('classic')
    plt.boxplot([df.duration[(df.error=='SMUX_OUT1') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT2') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT3') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT4') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT5') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT6') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT7') & (df.error_mode==type)],\
                 df.duration[(df.error=='SMUX_OUT8') & (df.error_mode==type)],\
                ], positions = range(1,9,1), widths = 0.6)
    # bp_min = plt.boxplot([Xe147,Xe133,Xe135], positions = [2,4,6], widths = 0.6)
    plt.yscale(yscale)
    plt.ylim(ylim)
    plt.grid()
    plt.xlabel('SMUX_OUT [channel]')
    plt.ylabel('duration [s]')
    for (i,j) in zip(np.arange(0.75,8.75,1),np.arange(1,9,1)):
        plt.text(i, 2e-9, 'n='+str(df.duration[df.error==('SMUX_OUT'+str(j))].count()))
    # for i in range(1,9,1):
    #     plt.xticks(range(1,9,1), label)
    plt.show()
    return [df,fig]

def plot_SET_count(filename=[],type='HighToLow'):
    #typ
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in typical condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        count_typ=list()
        df_typ= pd.DataFrame(index=[], columns=[])
        print 'Typ.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            count_typ.append(tmp.SET.id[tmp.SET.error_mode==type].count())
            df_typ=pd.concat([df_typ,tmp.SET],ignore_index=True)
    else:
        pass
    #min
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in minimum condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        count_min=list()
        df_min= pd.DataFrame(index=[], columns=[])
        print 'Min.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            count_min.append(tmp.SET.id[tmp.SET.error_mode==type].count())
            df_min=pd.concat([df_min,tmp.SET],ignore_index=True)
    else:
        pass
    error_min = np.sqrt(sum(count_min))
    error_typ = np.sqrt(sum(count_typ))
    label=['Typ','Min']
    fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
    plt.style.use('classic')
    plt.bar(1.5, sum(count_min), width=0.5, align='center',yerr=error_min, ecolor="black", capsize=10,label='min',alpha=0.5)
    plt.bar(0.5, sum(count_typ), width=0.5, align='center',yerr=error_typ, ecolor="black", capsize=10,label='typ',color='r',alpha=0.5)
    # plt.xlabel('SET width [us]')
    plt.ylabel('Counts')
    plt.xlim([0,2])
    plt.xticks([0.5,1.5], label)
    plt.grid()
    plt.legend(loc=0)
    plt.title('SET pulse counts')
    plt.show()

    fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
    plt.style.use('classic')
    bp_typ = plt.boxplot([df_typ.duration[df_typ.error_mode==type]], positions = [0.5], widths = 0.6)
    bp_min = plt.boxplot([df_min.duration[df_min.error_mode==type]], positions = [1.5], widths = 0.6)
    ## change outline color, fill color and linewidth of the boxes
    for box in bp_typ['boxes']:
        # change outline color
        box.set( color='r', linewidth=1)
        # change fill color
    #     box.set( facecolor = 'w' )
    ## change color and linewidth of the whiskers
    for whisker in bp_typ['whiskers']:
        whisker.set(color='r', linewidth=1)
    ## change color and linewidth of the caps
    for cap in bp_typ['caps']:
        cap.set(color='r', linewidth=1)
    ## change color and linewidth of the medians
    for median in bp_typ['medians']:
        median.set(color='r', linewidth=1)

    for box in bp_min['boxes']:
    # change outline color
        box.set( color='b', linewidth=1)
    # change fill color
    #     box.set( facecolor = 'w' )
    ## change color and linewidth of the whiskers
    for whisker in bp_min['whiskers']:
        whisker.set(color='b', linewidth=1)
    ## change color and linewidth of the caps
    for cap in bp_min['caps']:
        cap.set(color='b', linewidth=1)
    ## change color and linewidth of the medians
    for median in bp_min['medians']:
        median.set(color='b', linewidth=1)

    plt.xticks([0.5,1.5], label)
    # plt.xlabel('SET width [us]')
    plt.ylabel('SET width [u]')
    plt.ylim([1e-9,1e-6])
    plt.xlim([0,2])
    plt.yscale('log')
    plt.grid()
    plt.title('SET pulse width')
    plt.text(0.45, 2e-9, 'n='+str(df_typ.duration[df_typ.error_mode==type].count()))
    plt.text(1.45, 2e-9, 'n='+str(df_min.duration[df_min.error_mode==type].count()))
    plt.show()
    return [count_typ,count_min]

def plot_SET_interval(filename=[],type='HighToLow'):
    #typ
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in typical condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        count_typ=list()
        df_typ= pd.DataFrame(index=[], columns=[])
        print 'Typ.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            count_typ.append(tmp.SET.id[tmp.SET.error_mode==type].count())
            df_typ=pd.concat([df_typ,tmp.SET],ignore_index=True)
    else:
        pass
    #min
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in minimum condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        count_min=list()
        df_min= pd.DataFrame(index=[], columns=[])
        print 'Min.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            count_min.append(tmp.SET.id[tmp.SET.error_mode==type].count())
            df_min=pd.concat([df_min,tmp.SET],ignore_index=True)
    else:
        pass

    interval_typ=df_typ.time[df_typ.error_mode==type].dropna().sort_values().diff().astype('timedelta64[s]').dropna()
    interval_min=df_min.time[df_min.error_mode==type].dropna().sort_values().diff().astype('timedelta64[s]').dropna()

    fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
    plt.style.use('classic')
    bins=np.linspace(0,50,50)
    plt.hist(interval_typ,bins=bins,label='typ',color='r',alpha=0.5)
    plt.hist(interval_min,bins=bins,label='min',color='b',alpha=0.5)

    plt.ylabel('Counts')
    plt.xlabel('SET interval [s]')
    # plt.title('Hard-via-FPGA@CLK=10 MHz,input=0.01 Hz')
    plt.grid()
    plt.legend()
    plt.show()

def plot_SET_cross_section(filename=[],type='HighToLow'):
    #typ
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in typical condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        df_typ= pd.DataFrame(index=[], columns=[])
        Xsec_typ=list()
        print 'Typ.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            df_typ=pd.concat([df_typ,tmp.SET],ignore_index=True)
            Xsec_typ.extend((tmp.SET.time[tmp.SET.error_mode==type].dropna().sort_values().diff().astype('timedelta64[s]').dropna()*tmp.irradiation.flux.values[0])**-1)
    else:
        pass
    #min
    if not(filename):
        filenames=easygui.fileopenbox(msg='Open .mydata to plot in minimum condition.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    else:
        pass

    if filenames:
        df_min= pd.DataFrame(index=[], columns=[])
        Xsec_min=list()
        print 'Min.'
        for file in filenames:
            tmp=load_data(filename=file)
            print tmp.irradiation.ion.values[0]+'{0:03d}'.format(tmp.irradiation.number.values[0])
            df_min=pd.concat([df_min,tmp.SET],ignore_index=True)
            Xsec_min.extend((tmp.SET.time[tmp.SET.error_mode==type].dropna().sort_values().diff().astype('timedelta64[s]').dropna()*tmp.irradiation.flux.values[0])**-1)
    else:
        pass

    fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
    plt.style.use('classic')
    plt.subplot2grid((1,3),(0,0),colspan=2)
    plt.boxplot([Xsec_min,Xsec_typ],positions=[2,1],widths=0.5)
    plt.yscale('log')
    plt.ylim([1E-8,1E-4])
    plt.grid()
    # plt.xlabel('$LET\ [MeV/(cm^2/mg)]$')
    # plt.xlabel('Hard-via-FPGA@CLK=10 MHz,input=0.01 Hz')
    plt.ylabel('$SET\ cross\ section\\rm[cm^2/device]$')
    # plt.xticks(range(0,90,20),range(0,90,20))
    plt.xticks([1,2],['typ n='+str(len(Xsec_typ)),'min n='+str(len(Xsec_min))])
    plt.xlim([0,3])
    plt.tight_layout()

    plt.subplot2grid((1,3),(0,2))
    plt.hist(Xsec_min,label='min',bins=np.logspace(-8,-4,num=50),normed=False,color='b',alpha=0.5,orientation='horizontal')
    plt.hist(Xsec_typ,label='typ',bins=np.logspace(-8,-4,num=50),normed=False,color='r',alpha=0.5,orientation='horizontal')
    plt.grid()
    plt.yscale('log')
    plt.ylim([1E-8,1E-4])
    plt.legend()
    plt.tight_layout()
    plt.show()
    return [Xsec_min,Xsec_typ]

def fig2svg(fig,path=u'C:/Users/14026/Desktop/',format='svg'):
    fig.savefig(path+'image'+'.'+format,format=format)
    print('Saving... '+format+' format at'+path)

def pulse_test(filenames=[]):
    duration = pd.DataFrame(index=[''], columns=[])
    if not(filenames):
        filenames=easygui.fileopenbox(msg='Select waveform data.', title=None, default='*', filetypes=['*.csv'], multiple=True)
    else:
        pass
    if filenames:
        for filename in filenames:
            print filename.encode('utf-8')
            names=['t','XOR','CLK','SMUX_IN','SMUX_OUT8','SMUX_OUT7','SMUX_OUT6','SMUX_OUT5','SMUX_OUT4','SMUX_OUT3','SMUX_OUT2','SMUX_OUT1']
            df=pd.read_csv(filename,sep='\t',skiprows=22,names=names)
            if len(df.index) == 0:
                print u'blank file!'
                error_name=np.nan
                duration=np.nan
                error_mode=np.nan
            else:
                dt0=8.000000E-10 #delta t for XOR
                dt1=2.000000E-9  #delta t for degital signal
                df.t=df.index*dt0*10**6  #us
                df.t1=df.index*dt1*10**6 #us
                tmp = pd.DataFrame(index=[''], columns=[])
                if df.SMUX_OUT8.mean()<0.5:

                    tmp['SMUX_IN'] = df['SMUX_IN'][df['SMUX_IN']==1].count()*dt1
                    tmp['SMUX_OUT1'] = df['SMUX_OUT1'][df['SMUX_OUT1']==1].count()*dt1
                    tmp['SMUX_OUT2'] = df['SMUX_OUT2'][df['SMUX_OUT2']==1].count()*dt1
                    tmp['SMUX_OUT3'] = df['SMUX_OUT3'][df['SMUX_OUT3']==1].count()*dt1
                    tmp['SMUX_OUT4'] = df['SMUX_OUT4'][df['SMUX_OUT4']==1].count()*dt1
                    tmp['SMUX_OUT5'] = df['SMUX_OUT5'][df['SMUX_OUT5']==1].count()*dt1
                    tmp['SMUX_OUT6'] = df['SMUX_OUT6'][df['SMUX_OUT6']==1].count()*dt1
                    tmp['SMUX_OUT7'] = df['SMUX_OUT7'][df['SMUX_OUT7']==1].count()*dt1
                    tmp['SMUX_OUT8'] = df['SMUX_OUT8'][df['SMUX_OUT8']==1].count()*dt1
                else:
                    tmp['error_mode']='HighToLow'
                    tmp['SMUX_IN'] = df['SMUX_IN'][df['SMUX_IN']==0].count()*dt1
                    tmp['SMUX_OUT1'] = df['SMUX_OUT1'][df['SMUX_OUT1']==0].count()*dt1
                    tmp['SMUX_OUT2'] = df['SMUX_OUT2'][df['SMUX_OUT2']==0].count()*dt1
                    tmp['SMUX_OUT3'] = df['SMUX_OUT3'][df['SMUX_OUT3']==0].count()*dt1
                    tmp['SMUX_OUT4'] = df['SMUX_OUT4'][df['SMUX_OUT4']==0].count()*dt1
                    tmp['SMUX_OUT5'] = df['SMUX_OUT5'][df['SMUX_OUT5']==0].count()*dt1
                    tmp['SMUX_OUT6'] = df['SMUX_OUT6'][df['SMUX_OUT6']==0].count()*dt1
                    tmp['SMUX_OUT7'] = df['SMUX_OUT7'][df['SMUX_OUT7']==0].count()*dt1
                    tmp['SMUX_OUT8'] = df['SMUX_OUT8'][df['SMUX_OUT8']==0].count()*dt1
                duration=pd.concat([duration, tmp], ignore_index=True)
        duration=duration.dropna()
        return duration

def scanfolder(dir=[],extension='.csv'):
    df=pd.DataFrame(index=[], columns=['file'])
    if dir:
        pass
    else:
        dir=filename=easygui.diropenbox()
    for path, dirs, files in tqdm_notebook(os.walk(dir)):
        for f in files:
            if f.endswith(extension):
                # print os.path.join(path, f)
                tmp= pd.DataFrame({ 'file' : os.path.join(path, f) },index=[1])
                df=df.append(tmp,ignore_index=True)
    return df

def energy2LET(energy=1000,R=0.3539,d=0.035,l=0.0001):
    # R=0.3539
    E0=energy    #[pJ]
    Ep=3.6     #[eV]??
    Er=1.17    #[eV]
    alpha=14.8 #[cm-1]
    # d=0.035     #[cm]
    # l=0.0001   #[cm]
    rho=2330   #[mg/cm3]
    q=1.602e-19#[ev/J]
    LET=(1-R)*E0*1e-12/q/Er/Ep*np.exp(-alpha*d)*(1-np.exp(-alpha*l))/rho/l*1e-6 #[MeV/(mg/cm2)]
    return LET

def fit_weibull(LET,cross_section,A,x0,W,S,option=False):
    def Weibull(x,A,x0,W,S):
        return A*(1-np.exp(-((x-x0)/W)**S))

    model=lmfit.Model(Weibull)
    params = model.make_params()
    params.add('A', value=A, min=0, max=np.inf)
    params.add('x0', value=x0, min=0, max=np.inf)
    params.add('W', value=W, min=0, max=np.inf)
    params.add('S', value=S, min=0, max=np.inf)
    fit = model.fit(cross_section, params, x=LET ,fit_kws={'nan_policy':'omit'})
    print fit.fit_report()
    if option:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        plt.style.use('classic')
        plt.plot(LET,cross_section,'ro')
        t=np.arange(0,100,0.1)
        plt.plot(t,Weibull(t,**fit.values),'b')
        plt.grid()
        plt.yscale('log')
        # plt.xscale('log')
        plt.xlim([0,100])
        # plt.ylim([1e-10,1e-7])
        plt.ylabel(r'$cross\ section\ [cm^2/bit]$')
        plt.xlabel(r'$LET\ [MeV/(mg/cm^2)]$')
        plt.show()
    return fit

def digital_filter(data,dt,fcutoff,n=1,type='bessel',response=False,show=False,fft=False):
    #data   : filtering data  [-]
    #dt     : time resolution [s]
    #fcutoff: Cutoff frequency[Hz]
    #type   : filter type, 'bessel','butter'
    #n      : filter order    [-]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html#scipy.signal.bessel
    # https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.freqz.html#scipy.signal.freqz

    N = len(data)            # sample
    t = np.arange(0, N*dt, dt) # time
    fn=1.0/dt/2 #nyqfreq
    Ws = fcutoff/fn #normalize
    if type=='bessel':
        b, a = signal.bessel(n, Ws, 'low', analog=False,norm='mag') #bessel filter
    elif type=='butter':
        b, a = signal.butter(n, Ws, 'low')
    y = signal.filtfilt(b, a, data) #filter
    w,h=signal.freqz(b,a,worN=100000,whole=True) #freqztakeu

    if response:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        plt.style.use('classic')
#         plt.axhline(-3,ls='--',color='k')
        plt.axvline(fcutoff,ls='--',color='k')
        plt.plot(w/np.pi*fn, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.ylim([-50,0])
        plt.xlabel('Frequency [Hz]')
        ax1=plt.twinx()
        ax1.plot(w/np.pi*fn, np.unwrap(np.angle(h)), 'r')
        ax1.set_ylabel('Phase [degree]', color='r')
        ax1.set_ylim(top=0)
        plt.xscale('log')
#         plt.grid()
        plt.gca().set_xlim(right=fn)
        plt.show()

    if show:
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        plt.style.use('classic')
        plt.plot(t,data,'b-',alpha=0.5)
#         plt.ylim([-0.1,0.1])
#         plt.xlim([-20,60])
        plt.grid()
        plt.plot(t,y,'r-')
        plt.ylabel('Data')
        plt.xlabel('Time [s]')
        plt.show()

    if fft:
        F = np.fft.fft(y) #fft
        Amp = np.abs(F) #amplitude
        Pow = Amp ** 2 #power
        freq = np.linspace(0, 1.0/dt, N) #frequency
        fig=plt.figure(facecolor ="#FFFFFF",figsize=(8,5))
        plt.style.use('classic')
        plt.plot(freq, Pow,'r')
        plt.plot(freq,np.abs(np.fft.fft(data))**2,'b',alpha=0.5)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().set_xlim(right=fn)
        plt.axvline(fcutoff,ls='--',color='k')
        plt.grid()
        plt.show()

    return [y,t,b,a,w,h] #output

    def laser2LET(FWHM,E0th,E0,R=0.3539,d=0.03,l=0.00001,type='fixed_Buchner'):
        """
        Laserエネルギー->LET変換

        Parameters
        ----------
        FWHM : int or float
            パルスレーザ焦点面での半値全幅[um] (使用せず)
        E0th : int or float
            当該デバイスでSEUが発生する閾値[pJ]
        E0 : arrays of float or int
            LETに変換したいLaserエネルギー[pJ]
        R : float, default 0.3539
            反射率[無次元]
            デフォルト値=0.3539
        d : float, default 0.03
            感応領域までのSi基板厚み
            デフォルト値=0.03[cm] (300um)
        l :float, default 0.00001
            感応層厚み
            デフォルト値=0.00001[cm] (100nm)
        type : str, default 'fixed_Buchner'
            計算方式。デフォルト値='fixed_Buchner'
                'Buchner':
                    Buchner式
                    Buchner, S. P., Miller, F., Pouget, V., McMorrow, D. P. and Member, S. (2013) ‘Pulsed-Laser Testing for Single Event Effects Investigations’, IEEE Transactions on Nuclear Science, 60(3), pp. 1852–1875.

                'fixed_Buchner':
                    竹内-行松修正Buchner式

                'threshold':
                    閾値を考慮した竹内-行松修正Buchner式

        Returns
        -------
        LET : arrays of float or int
            LET。

        Notes
        -----

        """
        Ep=3.6     #[eV]
        Er=1.17    #[eV]Siのバンドギャップ
        alpha=14.8 #[cm-1]Siの透過率@1064nm
        rho=2330   #[mg/cm3]Si密度
        q=1.602e-19#[ev/J]素電荷

        sigma=FWHM/2/np.sqrt(2*np.log(2))
        if type=='fixed_Buchner':
            LET=(1-R)*E0*1e-12/q/Er/Ep*np.exp(-alpha*d)*(1-np.exp(-alpha*l))/rho/l*1e-6
        elif type=='Buchner':
            LET=(1-R)*E0*1e-12/q/Er/Ep*np.exp(-alpha*d)/rho/d*1e-6
        elif type=='threshold':
            LET=(1-R)*(E0-E0th)*1e-12/q/Er/Ep*np.exp(-alpha*d)*(1-np.exp(-alpha*l))/rho/l*1e-6
        else:
            LET=0
    return LET
