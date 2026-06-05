'''
plot mean IBI body angle distribution and standard deviation.

- change the var DAY_RESAMPLE & NIGHT_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change them to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import sys
import os,glob
from matplotlib import style
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid2
from scipy.stats import tukey_hsd

##### Parameters to change #####
pick_data = 'sldp' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' 'night', or 'all'
if_jackknife = False # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'IBI1_pitchMean2SDPercent_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI1_pitchMean2SDPercent_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')
defaultPlotting()
set_font_type()
# %%
# main function
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

bins = list(range(-90,95,5))

IBI_angles, all_cond0, all_cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','ztime','expNum','cond0','cond1','exp','propBoutIEI_pauseDur']]
IBI_angles_cond.columns = ['IBI_pitch','ztime','expNum','cond0','cond1','exp','dur']
IBI_angles_cond = IBI_angles_cond.dropna()
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

# Calculate std:
# std_jackknife v= IBI_toplt.groupby(cond_cols).apply(
    #     lambda x: jackknife_mean(x)
    # )
    # std_jackknife = std_jackknife.loc[:,'IBI_pitch'].reset_index()
    
# %%
# sanity check
# cat_cols = ['cond0','cond1','ztime','exp']
# check_df = IBI_angles.groupby(cat_cols).size().reset_index()
# check_df.columns = ['cond0','cond1','ztime','exp','bout_num']
# sns.catplot(data=check_df,x='exp',y='bout_num',col='ztime',row='cond0',hue='cond1',kind='bar')

# %% jackknife for day bouts
resampled_night_std = pd.DataFrame()
resampled_day_std = pd.DataFrame()

if if_jackknife:
    if which_ztime != 'night':
        IBI_angles_day_resampled = IBI_angles_cond.loc[
            IBI_angles_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                    ['cond0','cond1','exp']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        cat_cols = ['cond1','cond0','ztime']
        for (this_cond1, this_cond0, this_ztime), group in IBI_angles_day_resampled.groupby(cat_cols):
            expNum = group['expNum'].max()
            index_matrix = jackknife_resampling(np.array(list(range(expNum+1))))
            
            for excluded_exp, idx_group in enumerate(index_matrix):
                this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean().to_frame(name='IBIpitch_mean')
                this_sd = group.loc[group['expNum'].isin(idx_group),'IBI_pitch'].std()
                resampled_day_std = pd.concat([resampled_day_std, this_mean.assign(cond0=this_cond0,
                                                                        cond1=this_cond1,
                                                                        expNum=excluded_exp,
                                                                        ztime=this_ztime,
                                                                        IBIpitch_sd=this_sd)])
        resampled_day_std = resampled_day_std.reset_index(drop=True)


    if which_ztime != 'day':
        IBI_angles_night_resampled = IBI_angles_cond.loc[
            IBI_angles_cond['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            IBI_angles_night_resampled = IBI_angles_night_resampled.groupby(
                    ['cond0','cond1','exp']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True
                            )
        cat_cols = ['cond1','cond0','ztime']
        for (this_cond1, this_cond0, this_ztime), group in IBI_angles_night_resampled.groupby(cat_cols):
            expNum = group['expNum'].max()
            index_matrix = jackknife_resampling(np.array(list(range(expNum+1))))
            for excluded_exp, idx_group in enumerate(index_matrix):
                this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].median().to_frame(name='IBIpitch_mean')
                this_sd = group.loc[group['expNum'].isin(idx_group),'IBI_pitch'].std()
                resampled_night_std = pd.concat([resampled_night_std, this_mean.assign(cond0=this_cond0,
                                                                        cond1=this_cond1,
                                                                        expNum=excluded_exp,
                                                                        ztime=this_ztime,
                                                                        IBIpitch_sd=this_sd)])
        resampled_night_std = resampled_night_std.reset_index(drop=True)

    resampled_std = pd.concat([resampled_day_std,resampled_night_std]).reset_index(drop=True)
    IBI_toplt = resampled_std
    
else:
    IBI_toplt = IBI_angles_cond.groupby(['ztime','cond0','cond1','exp','expNum']).median().reset_index()
    IBI_toplt = IBI_toplt.assign(IBIpitch_mean = IBI_toplt['IBI_pitch'])
    IBI_toplt = IBI_toplt.assign(IBIpitch_sd = IBI_angles_cond.groupby(['ztime','cond0','cond1','exp','expNum'])['IBI_pitch'].std().values,
                                 )
IBI_toplt = IBI_toplt.assign(
    IBIpitch_mean2sd = IBI_toplt['IBIpitch_sd'] * 2 + IBI_toplt['IBIpitch_mean']
)
# IBI_std_day_resampled = IBI_angles_day_resampled.groupby(['ztime','cond0','cond1','expNum']).std().reset_index()

# %%
####################################
###### Plotting Starts Here ######
####################################

# plot kde of all
g = sns.FacetGrid(IBI_angles_cond, 
                  row="ztime", row_order=all_ztime,
                  col='cond0', col_order=all_cond0,
                  hue='cond1', hue_order=all_cond1,
                  )
g.map(sns.kdeplot, "IBI_pitch",alpha=0.5,)
g.add_legend()
filename = os.path.join(fig_dir,"IBI pitch kde.pdf")
plt.savefig(filename,format='PDF')

# %%


# %%
# %%
# pitch cond vs ctrl

IBI_toplt_wide = IBI_toplt.pivot(index=['cond0', 'expNum','ztime'], columns='cond1', values='IBIpitch_sd').reset_index()
IBI_toplt_wide = IBI_toplt_wide.assign(
    IBIpitch_std_percent = (IBI_toplt_wide[all_cond1[1]] - IBI_toplt_wide[all_cond1[0]])/IBI_toplt_wide[all_cond1[0]] * 100
)

if if_jackknife:
    prename = 'jackknife'
else:
    prename = ''

for feature in ['IBIpitch_std_percent']:
    x_name = 'cond0'
    gridrow = 'ztime'
    gridcol = None
    units = 'expNum'
    g = sns.catplot(
        data = IBI_toplt_wide,
        kind='point',
        markers='d',
        x = x_name,
        y = feature,
        row = gridrow,
        col = gridcol,
        errorbar='se',
        sharey=True,
        height = 3,
        aspect = 0.8,
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}_{x_name}__{gridcol}X{gridrow}_point.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %%
IBI_toplt_wide = IBI_toplt.pivot(index=['cond1', 'expNum','ztime'], columns='cond0', values='IBIpitch_sd').reset_index()
IBI_toplt_wide = IBI_toplt_wide.assign(
    IBIpitch_std_percent = (IBI_toplt_wide['DD'] - IBI_toplt_wide['LD'])/IBI_toplt_wide['LD'] * 100
)

if if_jackknife:
    prename = 'jackknife'
else:
    prename = ''

for feature in ['IBIpitch_std_percent']:
    x_name = 'cond0'
    gridrow = 'ztime'
    gridcol = None
    units = 'expNum'
    g = sns.catplot(
        data = IBI_toplt_wide,
        kind='point',
        markers='d',
        x = x_name,
        y = feature,
        row = gridrow,
        col = gridcol,
        errorbar='se',
        sharey=True,
        height = 3,
        aspect = 0.8,
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}_{x_name}__{gridcol}X{gridrow}_point.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %%
