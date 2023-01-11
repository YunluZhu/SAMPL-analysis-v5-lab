#%%
import sys
from plot_functions.plt_tools import round_half_up
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (defaultPlotting, set_font_type, plot_pointplt)
from plot_functions.get_IBIangles import get_IBIangles
from statsmodels.stats.multicomp import MultiComparison



def Fig7_bkg_IBI(root):
    defaultPlotting()
    set_font_type()
    # Paste root directory here
    which_zeitgeber = 'day'
    DAY_RESAMPLE = 0

    # %%

    print("- Figure 7: ZF strains - IBI")

    FRAME_RATE = 166
    
    folder_name = f'IBI features'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass

    # %%
    # main function
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    IBI_angles, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
    IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','propBoutIEI','ztime','expNum','condition0','condition','exp']]
    IBI_angles_cond.columns = ['IBI_pitch','IBI','ztime','expNum','condition0','condition','exp']
    IBI_angles_cond.reset_index(drop=True,inplace=True)
    cond_cols = ['ztime','condition0','condition']
    all_ztime = list(set(IBI_angles_cond.ztime))
    all_ztime.sort()
    

    night_std = pd.DataFrame()
    day_std = pd.DataFrame()

    if which_zeitgeber != 'night':
        IBI_angles_day_resampled = IBI_angles_cond.loc[
            IBI_angles_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                    ['condition0','condition','exp']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        cat_cols = ['condition','condition0','ztime']
        for (this_cond, this_condition0, this_ztime), group in IBI_angles_day_resampled.groupby(cat_cols):
            rep_list = group['expNum'].unique()
            for expNum in rep_list:
                this_std = group.loc[group['expNum'] == expNum,['IBI_pitch']].std().to_frame(name='IBI_std')
                this_mean = group.loc[group['expNum'] == expNum,['IBI_pitch']].mean()
                this_IBI = group.loc[group['expNum'] == expNum,['IBI']].mean()
                day_std = pd.concat([day_std, this_std.assign(condition0=this_condition0,
                                                                        condition=this_cond,
                                                                        expNum=expNum,
                                                                        ztime=this_ztime,
                                                                        mean_val=this_mean.values,
                                                                        IBI_val = this_IBI.values)])
        day_std = day_std.reset_index(drop=True)
        

    IBI_std = pd.concat([day_std,night_std]).reset_index(drop=True)

    coef_columns = ['IBI pitch std (deg)', 'IBI pitch (deg)', 'IBI duration (s)']
    IBI_std = IBI_std.loc[:,['IBI_std','mean_val','IBI_val','condition']]
    IBI_std.columns = coef_columns + ['condition']
    
    # plot IBI features

    for i, coef_col_name in enumerate(coef_columns):
        plot_pointplt(IBI_std,coef_col_name,cond2)
        filename = os.path.join(fig_dir,f"{coef_col_name} .pdf")
        plt.savefig(filename,format='PDF')
        
        print(f"{coef_col_name}: ")
        multi_comp = MultiComparison(IBI_std[coef_col_name], IBI_std['condition'])
        print(multi_comp.tukeyhsd().summary())

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')    
    Fig7_bkg_IBI(root)