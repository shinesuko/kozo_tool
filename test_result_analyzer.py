#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import easygui

filename=kt.create_master_data()
kt.create_irradiation_data(filename=filename)

filenames=easygui.fileopenbox(msg='Open mydata to add current data.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
for filename in filenames:
    kt.add_current_data(filename)

filenames=easygui.fileopenbox(msg='Open mydata to add SEE data.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
for filename in filenames:
    kt.add_SEE_data(filename)

print('END')
