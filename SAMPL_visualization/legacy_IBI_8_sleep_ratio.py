'''
Plot bout features - UP DOWN separated by set point

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknnife = False # Whether to jackknife (bool)

- change the var DAY_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change DAY_RESAMPLE to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True

'''

#%%
# import sys
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.plt_tools import (set_font_type, defaultPlotting,boot_ci)
from plot_functions.plt_functions import plt_categorical_combined
from functools import partial
import matplotlib as mpl
from scipy.stats import ttest_rel, norm, median_test
import random


# %%
##### Parameters to change #####
pick_data = 'a_gtau'
which_ztime = 'all' # 'day', 'night', or 'all'
if_boot = True # Whether to bootstrap

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'IBI8_sleepRatio_{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %% get features
IBI_angles, all_cond0, all_cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','propBoutIEI_angVel_postBout', 'propBoutIEI_angVel_preNextBout', 'propBoutIEI_angVel','propBoutIEI_pauseDur','ztime','expNum','cond0','cond1','exp']]
IBI_angles_cond.columns = ['pitch','IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = IBI_angles_cond.ztime.unique()
all_ztime.sort()

sel_features = ['yvel','pitch','IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', '','IBI_pauseDur']
# Calculate std:# %% tidy data
all_feature_cond = IBI_angles_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# %%
# assign up and down 

all_feature_toplt = all_feature_cond.copy()


# %%
plt_median_ci = partial(boot_ci, stat='median', boot_rep=100, confidence_level=0.99)

feature_to_plt = ['IBI_pauseDur']

sleep_threshold = 8


toplt = all_feature_toplt.query("IBI_pauseDur > 0")
toplt= all_feature_toplt.query("IBI_pauseDur < @sleep_threshold")
for feature in feature_to_plt:
    g = sns.catplot(
        data=toplt,
        x='cond1',
        col='ztime',
        row='cond0',
        y=feature,
        kind='point',
        estimator=np.nanmedian,
        errorbar=plt_median_ci,
        height=3,
        sharey=True
    )
    for ztime in ['day', 'night']:
        this_compare = toplt.query('ztime == @ztime')
        print(ztime)
        mres = median_test(this_compare.loc[this_compare['cond1']==all_cond1[0],feature].values,
                                this_compare.loc[this_compare['cond1']==all_cond1[1],feature].values)
        print(f'{feature} Sibs v.s. Tau: paired median p-value = {mres.pvalue}')
    plt.savefig(fig_dir+f"/{feature} point.pdf",format='PDF')
    

toplt = all_feature_toplt.query("IBI_pauseDur > @sleep_threshold")
for feature in feature_to_plt:
    g = sns.catplot(
        data=toplt,
        x='cond1',
        col='ztime',
        row='cond0',
        y=feature,
        kind='point',
        estimator=np.nanmedian,
        errorbar=plt_median_ci,
        height=3
    )
    for ztime in ['day', 'night']:
        this_compare = toplt.query('ztime == @ztime')
        print(ztime)
        mres = median_test(this_compare.loc[this_compare['cond1']==all_cond1[0],feature].values,
                                this_compare.loc[this_compare['cond1']==all_cond1[1],feature].values)
        print(f'{feature} Sibs v.s. Tau: paired median p-value = {mres.pvalue}')
    plt.savefig(fig_dir+f"/{feature} point sleep only.pdf",format='PDF')

# %% bootstrap

BOOT_REP = 50
boot_dfall_ = []
for (cond0, cond1, ztime), group in all_feature_toplt.groupby(['cond0','cond1','ztime']):
    bst_input = group.index
    if BOOT_REP > 1:
        bst_output = [
            np.array(random.choices(bst_input, k=len(bst_input)))
            for i in np.arange(BOOT_REP)
        ]
        for bts, ind in enumerate(bst_output):
            bootdf = group.loc[ind, :]
            bootdf = bootdf.assign(
                bts_rep = bts
            )
            boot_dfall_.append(bootdf)
boot_dfall = pd.concat(boot_dfall_, ignore_index=True)

#%%

sleep_threshold = 8


if if_boot:
    data_toplt = boot_dfall.dropna()
    rep_col = 'bts_rep'
else:
    data_toplt = all_feature_toplt.dropna()
    rep_col = 'expNum'
left = data_toplt.query("IBI_pauseDur > @sleep_threshold").groupby(['cond0','cond1','ztime',rep_col]).size().reset_index()
right = data_toplt.groupby(['cond0','cond1','ztime',rep_col]).size().reset_index()
count = left.merge(right, on=['cond0','cond1','ztime',rep_col])
count.columns = ['cond0','cond1','ztime',rep_col, 'sleep', 'all']
count = count.assign(
    ratio_of_sleep = count['sleep']/count['all'] * 100
)
# %
count = count.sort_values(by=['cond0','cond1','ztime'])
feature = 'ratio_of_sleep'
g = plt_categorical_combined(
    data=count,
    x='cond1',
    col='ztime',
    row='cond0',
    y=feature,
    units=rep_col,
    height=3,
    overlay_func=sns.stripplot,
    alpha=0.1,
    estimator=np.nanmean,
    errorbar='sd',
    sharey=True
)
for ztime in ['day', 'night']:
    this_compare0 = count.query('ztime == @ztime')
    print(ztime)
    for cond0 in count.cond0.unique():
        this_compare = this_compare0.query('cond0 == @cond0')
        print(cond0)
        ctrl_df = this_compare.loc[this_compare['cond1']==all_cond1[0],feature].values
        cond_df = this_compare.loc[this_compare['cond1']==all_cond1[1],feature].values
        diff = cond_df - ctrl_df
        boot_mean, boot_std = np.mean(diff), np.std(diff)
        p_1 = norm.cdf(0, boot_mean, boot_std)
        p_2 = norm.cdf(0, -boot_mean, boot_std)
        p_value = min(p_1, p_2) * 2
        print(f"p_value: {p_value}")
        
plt.savefig(fig_dir+f"/{feature} of sleep point.pdf",format='PDF')

# %%
g = sns.displot(data=all_feature_toplt, x = 'IBI_pauseDur', col='ztime',hue='cond1' , kind='ecdf')
g.set(xlim=[0,20])


#%%
df_input = all_feature_toplt.dropna()
thresholdSleep = np.percentile(df_input.query('cond1=="1sibs"').IBI_pauseDur,80)
df_input = df_input.loc[df_input['IBI_pauseDur']< 45].reset_index(drop=True)

df_input.groupby(['cond0','cond1','ztime'])['IBI_pauseDur'].apply(lambda g: np.sum(g>thresholdSleep)/len(g))
# %%
