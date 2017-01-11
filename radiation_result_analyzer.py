#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import pandas as pd
import numpy as np
import easygui
import os
from datetime import datetime

#cd
os.chdir('C:/Users/14026/Desktop')
filename=easygui.fileopenbox(msg='Open master data.', title=None, default='*', filetypes=['*.csv'], multiple=False)

if filename:
    if os.path.isfile(os.path.dirname(filename)+'/master.mydata'):
        #open master data
        data=kt.load_data(os.path.dirname(filename)+'/master.mydata')
        print('Reading... '+os.path.dirname(filename).encode('utf-8')+'/master.mydata')
    else:
        #anonymous class
        data=kt.structure()
        data.version=1.0
        data.field=['version','field']

        #open master data
        print filename.encode('utf-8')
        master = pd.read_table(filename,delimiter=',',escapechar='&',header=0,index_col=False,parse_dates=[9,10,11,12],na_values=['NoData',' '])
        master.index.names = ['index']
        master.columns.names = ['column']
        data.master=master
        data.field.append('master')
        #save master.mydata
        kt.save_data(data,os.path.dirname(filename)+'/master.mydata')
        print('Saving... '+os.path.dirname(filename).encode('utf-8')+'/master.mydata')

    #set irradiation data
    #ion
    ion_choices = ['Xe', 'Kr', 'Ar', 'Ne','N','Cf' ,'Am']
    ion = easygui.choicebox('Choose ion.', '', ion_choices)
    #test number
    fieldValue = []  # we start with blanks for the values
    fieldValue = easygui.multenterbox('Enter test number in numeric character.','', ['test number'])

    if ion and fieldValue:
        formatted = '%03d' % int(''.join(fieldValue))
        print('Converting... '+ion+formatted)
        data.irradiation=data.master.loc[(data.master['ion']==ion) & (data.master['number']==int(''.join(fieldValue))),:]
        data.field.append('irradiation')
        print(data.irradiation)

        #open current data
        os.chdir(os.path.dirname(filename))
        os.chdir("..")
        print os.getcwd()
        filenames_current=easygui.fileopenbox(msg='Open '+ion+formatted+' current data.', title=None, default='*', filetypes=['*.txt'], multiple=True)
        if filenames_current:
            data.current=pd.DataFrame(index=[], columns=['DateTime'])
            data.field.append('current')
            print(data.current)
            for file in filenames_current:
                df,_=kt.read_current_data(filename=file)
                data.current=pd.merge(data.current,df,on='DateTime',how='outer')
            print(data.current)
        else:
            pass

        #open see data

        #timestamp
        data.timestamp=datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        data.field.append('timestamp')

        #save data
        kt.save_data(data,os.path.dirname(filename)+'/'+ion+formatted+'.mydata')
        print('Saving... '+os.path.dirname(filename).encode('utf-8')+'/'+ion+formatted+'.mydata')

    else:
        pass

else:
    print('Canceled!')

print(data.field)

os.chdir('C:/Users/14026/Desktop')
print('End of procedure.')
#EOF
