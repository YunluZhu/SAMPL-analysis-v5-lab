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
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_list)
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
    folder_name = f'IBI1_pitch_byBox_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI1_pitch_byBox_z{which_ztime}'
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
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','ztime','expNum','cond0','cond1','exp','propBoutIEI_pauseDur','boxNum']]
IBI_angles_cond.rename(columns={'propBoutIEI_pitch':"IBI_pitch",
                                'propBoutIEI_pauseDur':'dur'}, inplace=True)
IBI_angles_cond.reset_index(drop=True,inplace=True)
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()
IBI_angles_cond = IBI_angles_cond.assign(
    expBoxNum = IBI_angles_cond['expNum'].astype(str) + '_' + IBI_angles_cond['boxNum'].astype(str) 
)
cond_cols = ['ztime','cond0','cond1','boxNum','expBoxNum']

# Calculate std:
# std_jackknife v= IBI_std_cond.groupby(cond_cols).apply(
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
            jackknife_idx = jackknife_list(group['expBoxNum'].unique())            
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                this_std = group.loc[group['expBoxNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='IBIpitch_std')
                this_mean = group.loc[group['expBoxNum'].isin(idx_group),['IBI_pitch']].mean()
                resampled_day_std = pd.concat([resampled_day_std, this_std.assign(cond0=this_cond0,
                                                                        cond1=this_cond1,
                                                                        expBoxNum=excluded_exp,
                                                                        ztime=this_ztime,
                                                                        mean_val=this_mean)])
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
            jackknife_idx = jackknife_list(group['expBoxNum'].unique())            
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                this_std = group.loc[group['expBoxNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='IBIpitch_std')
                this_mean = group.loc[group['expBoxNum'].isin(idx_group),['IBI_pitch']].mean()
                resampled_night_std = pd.concat([resampled_night_std, this_std.assign(cond0=this_cond0,
                                                                        cond1=this_cond1,
                                                                        expBoxNum=excluded_exp,
                                                                        ztime=this_ztime,
                                                                        mean_val=this_mean)])
        resampled_night_std = resampled_night_std.reset_index(drop=True)

    resampled_std = pd.concat([resampled_day_std,resampled_night_std]).reset_index(drop=True)
    IBI_std_cond = resampled_std
    
else:
    IBI_std_cond = IBI_angles_cond.groupby(['ztime','cond0','cond1','expNum','expBoxNum']).std().reset_index()
    IBI_std_cond = IBI_std_cond.assign(IBIpitch_std = IBI_std_cond['IBI_pitch'])
    IBI_std_cond = IBI_std_cond.assign(mean_val = IBI_angles_cond.groupby(['ztime','cond0','cond1','expNum','expBoxNum'])['IBI_pitch'].mean().values)
IBI_std_cond = IBI_std_cond.assign(
    mean2sd = IBI_std_cond['mean_val'] + 2* IBI_std_cond['IBIpitch_std'],
    mean1sd = IBI_std_cond['mean_val'] + IBI_std_cond['IBIpitch_std']
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
# pitch cond vs ctrl

if if_jackknife:
    prename = 'jackknife'
else:
    prename = ''

for feature in ['IBIpitch_std', 'mean_val', 'mean2sd','mean1sd']:
    x_name = 'cond1'
    gridrow = 'ztime'
    gridcol = 'cond0'
    units = 'expBoxNum'
    g = plt_categorical_grid2(
        data = IBI_std_cond,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=False,
        height = 3,
        aspect = 0.8,
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %
    g = sns.catplot(
        data = IBI_std_cond,
        x = x_name,
        y = feature,
        row = gridrow,
        col = gridcol,
        hue = units,
        sharey=False,
        height = 3,
        aspect = 0.8,
        palette=sns.color_palette("tab10")
        )
    plt.show()

    condition2_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[2],feature]
    condition_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[1],feature]
    control_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[0],feature]
    res = tukey_hsd(control_df, condition_df, condition2_df)
    print(res)
# %%
sns.relplot(
    data = IBI_std_cond,
    hue = x_name,
    x = 'mean_val',
    y = 'IBIpitch_std',
    row = gridrow,
    col = gridcol,
    height = 3,
    aspect = 0.8,
)

#%%
plt.show()

condition2_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[2],feature]
condition_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[1],feature]
control_df = IBI_std_cond.loc[IBI_std_cond['cond1'] == all_cond1[0],feature]
res = tukey_hsd(control_df, condition_df, condition2_df)
print(res)
# %%
