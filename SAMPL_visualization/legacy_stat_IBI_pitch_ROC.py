#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.plt_stats import calc_ROC
from plot_functions.get_IBIangles import get_IBIangles
from scipy import stats

pick_data = 'tau_bkg'
which_ztime = 'day'
DAY_RESAMPLE = 1000  # how many bouts to take per  exp/ztime/condition
RESAMPLE = DAY_RESAMPLE

set_font_type()
defaultPlotting(size=16)
# %%

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'stat_ROC'
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

IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','ztime','expNum','cond0','cond1','exp']]
IBI_angles_cond.columns = ['IBI_pitch','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

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
# not the best code - jackknife and resample to be wrapped into a function

jackknifed_night_std = pd.DataFrame()
jackknifed_day_std = pd.DataFrame()

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
    for (this_cond, this_dpf, this_ztime), group in IBI_angles_day_resampled.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
            this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
            jackknifed_day_std = pd.concat([jackknifed_day_std, this_std.assign(dpf=this_dpf,
                                                                    cond1=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime,
                                                                    jackknifed_mean=this_mean)])
    jackknifed_day_std = jackknifed_day_std.reset_index(drop=True)


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
    for (this_cond, this_dpf, this_ztime), group in IBI_angles_night_resampled.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
            this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
            jackknifed_night_std = pd.concat([jackknifed_night_std, this_std.assign(dpf=this_dpf,
                                                                    cond1=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime,
                                                                    jackknifed_mean=this_mean)])
    jackknifed_night_std = jackknifed_night_std.reset_index(drop=True)

jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)
IBI_std_cond = IBI_angles_cond.groupby(['ztime','cond0','cond1','exp','expNum']).std().reset_index()
IBI_std_day_resampled = IBI_angles_day_resampled.groupby(['ztime','cond0','cond1','expNum']).std().reset_index()

# %%
# ROC

# %%
FPR_list, TPR_list, auc = calc_ROC(jackknifed_std,'jackknifed_std',cond1[0],'increase')  # left = cond is expected to be smaller than ctrl
# %%
fig, ax = plt.subplots(1,1, figsize=(3,3))

ax.plot(FPR_list, TPR_list)
ax.plot((0,1), "--")
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])
ax.set_title("ROC Curve", fontsize=14)
ax.set_ylabel('TPR', fontsize=12)
ax.set_xlabel('FPR', fontsize=12)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.legend([f"AUC = {np.mean(auc):.3f}±{np.std(auc):.3f}"])
filename = os.path.join(fig_dir,f"ROC_IBIstd_{which_ztime}_sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')

# %%
# paired ttest
for condition in cond1:
    df = jackknifed_std.loc[jackknifed_std.cond0 == condition]
    cond = df.loc[df.cond1==cond1[0],'jackknifed_std'].values
    ctrl = df.loc[df.cond1==cond1[1],'jackknifed_std'].values
    print(f'{condition} jackknifed_std cond-ctrl paired ttest')
    print(stats.ttest_rel(cond,ctrl))
# %%
