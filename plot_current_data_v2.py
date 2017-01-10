#coding: utf-8
#! D:\Anaconda3\envs\py27

import kozo_tool as kt
import mpld3
import easygui

filenames=easygui.fileopenbox(msg=None, title=None, default='*', filetypes=['*.txt'], multiple=True)

for filename in filenames:
    #read from filename
    df,filename=kt.read_current_data(filename=filename)
    #plot from dataframe
    fig,ax=kt.plot_figure(df,filename=filename,show=False)
    #save as html
    save_filename = open(filename[0:-4]+'.html', 'wb')
    mpld3.save_html(fig,save_filename)
    save_filename.close()
    #save as png
    fig.savefig(filename[0:-4]+'.png',format='png',dpi=300)
    #save as eps
    fig.savefig(filename[0:-4]+'.eps',format='eps',dpi=300)


#save
# print sys.getrecursionlimit()
# sys.setrecursionlimit(5000)
# print os.path.basename(filename[1:-4])
# print os.path.dirname(filename)
# save_filename = open(filename[0:-4]+'.pickle', 'wb')
# dill.dump(fig, save_filename)
# save_filename.close()
# kt.load_figure_pickle()
