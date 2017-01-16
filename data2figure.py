#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import easygui

filenames=easygui.fileopenbox(msg='Open mydata.', title=None, default='*', filetypes=['*.mydata'], multiple=True)
for filename in filenames:
    kt.add_figure(filename=filename,show=False,html=True,png=True,eps=False)
