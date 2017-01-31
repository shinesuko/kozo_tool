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


def read_current_data(filename=[]):
    if not(filename):
        filename=easygui.fileopenbox()
    else:
        pass
    if filename:
        print filename.encode('utf-8')
        try:
            df = pd.read_table(filename,delimiter='\t',escapechar='&',header=1,skipfooter=1,index_col=False,parse_dates=[0],na_values='NoData',engine='python')
            df.columns = ['DateTime','Elap[sec]','Ch1[I]','Ch2[I]']
            print('reading as 6581 file format.')
        except:
            # print('reading as 6629 file format.')
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
        os.chdir('..')
        print('cd to... '+os.getcwd())
        data=load_data(filename=filename)
        formatted = '%03d' % int(data.irradiation['number'])
        print type(data.irradiation['ion'])
        ion=data.irradiation['ion']
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
        ion=data.irradiation['ion']
        print(ion+formatted+'.mydata is seleced')
        filenames_SEE=easygui.fileopenbox(msg='Open '+ion+formatted+' SEE data.', title=None, default='*', filetypes=['*.jpg'], multiple=True)
        if filenames_SEE:
            data.SEE=pd.DataFrame(index=[], columns=['DateTime'])
            if 'SEE' in data.field:
                pass
            else:
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

def add_figure(filename=[],show=False,html=False,png=False,eps=False,add_data=False,fix_time=False):
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
            fig=plt.figure()
            fig.patch.set_facecolor('white')  # 図全体の背景色
            ax=plt.axes()
            plt.hold(True)
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

            plt.ylim([1E-5,1E-1])
            plt.yscale('log')
            plt.legend(loc='lower right',prop={'size':10})
            plt.xlabel('Time')
            plt.ylabel('Current [A]')
            plt.title(os.path.basename(filename).encode('utf-8'))
            plt.xticks(rotation=30)
            timeformat = mdates.DateFormatter('%H:%M:%S')
            ax.xaxis.set_major_formatter(timeformat)
            plt.tight_layout()  # タイトルの被りを防ぐ

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
                    plt.hold(True)
                    if fix_time:
                        plt.axvspan(xmin=data.irradiation.current_start.iloc[0],xmax=data.irradiation.current_end.iloc[0],facecolor='gray', alpha=0.1)
                    else:
                        print('fix_time:disabled')
                        plt.axvspan(xmin=data.irradiation.irradiation_start.iloc[0],xmax=data.irradiation.irradiation_end.iloc[0],facecolor='gray', alpha=0.1)
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

            data.fig=fig
            data.plt=plt
            data.ax=ax
            with open('C:/Users/14026/Desktop/test.mydata','wb') as f:
                dill.dump(data,f)

        else:
            print(filename.encode('utf-8')+' has no data to plot figure')
    else:
        pass
    print('End of add_figure.')

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
            print(data.irradiation['ion'].iloc[0], data.irradiation['number'].iloc[0])
            data.irradiation=data.master[(data.master['ion']==data.irradiation['ion'].iloc[0]) & (data.master['number']==data.irradiation['number'].iloc[0])]
            print data.irradiation
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

def data2figure(filename=[],show=False,html=True,png=True,eps=False,add_data=True,fix_time=True):
    print('End of... data2figure.')
    filenames=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    if filenames:
        for filename in filenames:
            add_figure(filename=filename,show=show,html=html,png=png,eps=eps,add_data=add_data,fix_time=fix_time)
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
