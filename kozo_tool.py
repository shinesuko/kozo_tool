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
            fig=plt.figure()
            fig.patch.set_facecolor('white')  # 図全体の背景色
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
                    plt.hold(True)
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

def data2figure(filename=[],show=False,html=True,png=True,eps=False,add_data=True,fix_time=True,ymin=1E-6,ymax=1E-1,yscale='log'):
    print('Starting... data2figure.')
    filenames=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
    if filenames:
        for filename in filenames:
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

def read_SRIM(thick=0.2):
    #すべてのSRIM outputには対応していない
    filename1=easygui.fileopenbox(msg='Open ion in Gold file', title=None, default='*', multiple=False)
    filename2=easygui.fileopenbox(msg='Open ion in Silicon file', title=None, default='*', multiple=False)
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

    return [df1,df2,df3,df4]

def SRIM():
    filename=easygui.fileopenbox()
    print filename.encode('utf-8')
    #すべてのSRIM outputには対応していない
    col_names = ['Energy','Energy_unit','de/dx_elec','dE/dx_nuc','Range','Range_unit','Longitudinal_straggling','Longitudinal_straggling_unit','Lateral_straggling','Lateral_straggling_unit','A']
    df=pd.read_csv(filename,sep=' ',header='infer',skiprows=24,skipinitialspace=True,skipfooter=13,names=col_names)
    del df['A']
    df['Energy'][df['Energy_unit']=='keV']=df['Energy'][df['Energy_unit']=='keV'].copy()/1000
    # df['Energy_unit'][df['Energy_unit']=='keV']='MeV'
    df['Range'][df['Range_unit']=='A']=df['Range'][df['Range_unit']=='A'].copy()/10000
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
    ax1.set_ylim([0,20])
    ax2.set_ylim([0,80])
    ax1.set_xlim([0,18])
    plt.title(os.path.basename(filename))
    plt.grid()
    plt.show()

    return df
