'''
plot mean IBI body angle distribution and standard deviation.

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - separated sample number for day and night
- change the var DAY_RESAMPLE & NIGHT_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change them to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import sys
import os,glob
from matplotlib import style
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling

from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_mean, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
defaultPlotting()
set_font_type()
# %%
# Paste root directory here
pick_data = 'tau_bkg'
which_zeitgeber = 'all'
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0

# %%
# ztime_dict = {}

root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'IBI_pitch_z{which_zeitgeber}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI_pitch_z{which_zeitgeber}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

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

IBI_angles, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','ztime','expNum','dpf','condition','exp']]
IBI_angles_cond.columns = ['IBI_pitch','ztime','expNum','dpf','condition','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','dpf','condition']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

# Calculate std:
# std_jackknife v= IBI_std_cond.groupby(cond_cols).apply(
    #     lambda x: jackknife_mean(x)
    # )
    # std_jackknife = std_jackknife.loc[:,'IBI_pitch'].reset_index()
    
# %%
# sanity check
cat_cols = ['dpf','condition','ztime','exp']
check_df = IBI_angles.groupby(cat_cols).size().reset_index()
check_df.columns = ['dpf','condition','ztime','exp','bout_num']
sns.catplot(data=check_df,x='exp',y='bout_num',col='ztime',row='dpf',hue='condition',kind='bar')

# %% jackknife for day bouts
# not the best code - jackknife and resample to be wrapped into a function

jackknifed_night_std = pd.DataFrame()
jackknifed_day_std = pd.DataFrame()

if which_zeitgeber != 'night':
    IBI_angles_day_resampled = IBI_angles_cond.loc[
        IBI_angles_cond['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                ['dpf','condition','exp']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )
    cat_cols = ['condition','dpf','ztime']
    for (this_cond, this_dpf, this_ztime), group in IBI_angles_day_resampled.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(IBI_angles_day_resampled['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
            this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
            jackknifed_day_std = pd.concat([jackknifed_day_std, this_std.assign(dpf=this_dpf,
                                                                    condition=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime,
                                                                    jackknifed_mean=this_mean)])
    jackknifed_day_std = jackknifed_day_std.reset_index(drop=True)


if which_zeitgeber != 'day':
    IBI_angles_night_resampled = IBI_angles_cond.loc[
        IBI_angles_cond['ztime']=='night',:
            ]
    if NIGHT_RESAMPLE != 0:  # if resampled
        IBI_angles_night_resampled = IBI_angles_night_resampled.groupby(
                ['dpf','condition','exp']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True
                        )
    cat_cols = ['condition','dpf','ztime']
    for (this_cond, this_dpf, this_ztime), group in IBI_angles_night_resampled.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(IBI_angles_night_resampled['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
            this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
            jackknifed_night_std = pd.concat([jackknifed_night_std, this_std.assign(dpf=this_dpf,
                                                                    condition=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime,
                                                                    jackknifed_mean=this_mean)])
    jackknifed_night_std = jackknifed_night_std.reset_index(drop=True)

jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)
IBI_std_cond = IBI_angles_cond.groupby(['ztime','dpf','condition','exp','expNum']).std().reset_index()
IBI_std_day_resampled = IBI_angles_day_resampled.groupby(['ztime','dpf','condition','expNum']).std().reset_index()

# %% ignore this

# %%
# plot kde of all
g = sns.FacetGrid(IBI_angles_cond, 
                  row="ztime", row_order=all_ztime,
                  col='dpf', col_order=cond1,
                  hue='condition', hue_order=cond2,
                  )
g.map(sns.kdeplot, "IBI_pitch",alpha=0.5,)
g.add_legend()
filename = os.path.join(fig_dir,"IBI pitch kde.pdf")
plt.savefig(filename,format='PDF')

# %% 
# raw pitch day vs night

if which_zeitgeber == 'all':
    plt.close()
    g = sns.catplot(data=IBI_std_cond,
                    col='dpf',row='condition',
                    x='ztime', y='IBI_pitch',
                    hue='dpf',
                    ci='sd',
                    kind='point')
    g.map(sns.lineplot,'ztime','IBI_pitch',estimator=None,
      units='expNum',
      data = IBI_std_cond,
      alpha=0.2,)
    filename = os.path.join(fig_dir,"pitch_IBI day-night.pdf")
    plt.savefig(filename,format='PDF')

# %%
# pitch cond vs ctrl

g = sns.catplot(data=IBI_std_cond,
                col='dpf',
                row='ztime',
                x='condition', y='IBI_pitch',
                hue='expNum',
                ci=None,
                # markers=['d','d'],
                sharey=False,
                kind='point',
                aspect=.5
                )
g.map(sns.lineplot,'condition','IBI_pitch',estimator=None,
      units='expNum',
      data = IBI_std_cond,
      alpha=0.2,)
filename = os.path.join(fig_dir,"pitch_IBI exp repeats.pdf")
plt.savefig(filename,format='PDF')


g = sns.catplot(data=IBI_std_cond,
                col='dpf',row='ztime',
                x='condition', y='IBI_pitch',
                hue='dpf',
                ci='sd',
                kind='point')
g.map(sns.lineplot,'condition','IBI_pitch',estimator=None,
    units='expNum',
    data = IBI_std_cond,
    alpha=0.2,)
filename = os.path.join(fig_dir,"pitch_IBI cond.pdf")
plt.savefig(filename,format='PDF')

# %%
# jackknifed resampled per repeat
# mean cond vs ctrl

plt.close()
g = sns.catplot(data=jackknifed_std,
                col='dpf',
                row='ztime',
                x='condition', y='jackknifed_mean',
                hue='condition',
                ci='sd', 
                # markers=['d','d'],
                sharey=False,
                kind='point',
                aspect=.8
                )
g.map(sns.lineplot,'condition','jackknifed_mean',estimator=None,
      units='excluded_exp',
      data = jackknifed_std,
      color='grey',
      alpha=0.2,)
g.add_legend()
sns.despine(offset=10)
filename = os.path.join(fig_dir,"std of IBI pitch - jackknifed resasmpled.pdf")
plt.savefig(filename,format='PDF')

# jackknifed resampled per repeat
# mean cond vs ctrl
# plot on same scale

plt.close()
g = sns.catplot(data=jackknifed_std,
                col='dpf',
                row='ztime',
                x='condition', y='jackknifed_mean',
                hue='condition',
                ci='sd', 
                # markers=['d','d'],
                sharey='row',
                kind='point',
                aspect=.8
                )
g.map(sns.lineplot,'condition','jackknifed_mean',estimator=None,
      units='excluded_exp',
      data = jackknifed_std,
      color='grey',
      alpha=0.2,)
g.add_legend()
sns.despine(offset=10)
filename = os.path.join(fig_dir,"std of IBI pitch - jackknifed sharey.pdf")
plt.savefig(filename,format='PDF')

# %%
# jackknifed resampled per repeat
# std cond vs ctrl

plt.close()
g = sns.catplot(data=jackknifed_std,
                col='dpf',
                row='ztime',
                x='condition', y='jackknifed_std',
                hue='condition',
                ci='sd', 
                # markers=['d','d'],
                sharey=False,
                kind='point',
                aspect=.8
                )
g.map(sns.lineplot,'condition','jackknifed_std',estimator=None,
      units='excluded_exp',
      data = jackknifed_std,
      color='grey',
      alpha=0.2,)
g.add_legend()
sns.despine(offset=10)
filename = os.path.join(fig_dir,"std of IBI pitch - jackknifed resasmpled.pdf")
plt.savefig(filename,format='PDF')

# jackknifed resampled per repeat
# std cond vs ctrl
# plot on same scale

plt.close()
g = sns.catplot(data=jackknifed_std,
                col='dpf',
                row='ztime',
                x='condition', y='jackknifed_std',
                hue='condition',
                ci='sd', 
                # markers=['d','d'],
                sharey='row',
                kind='point',
                aspect=.8
                )
g.map(sns.lineplot,'condition','jackknifed_std',estimator=None,
      units='excluded_exp',
      data = jackknifed_std,
      color='grey',
      alpha=0.2,)
g.add_legend()
sns.despine(offset=10)
filename = os.path.join(fig_dir,"std of IBI pitch - jackknifed sharey.pdf")
plt.savefig(filename,format='PDF')
# %%
