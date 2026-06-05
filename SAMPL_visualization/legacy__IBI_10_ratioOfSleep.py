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
from plot_functions.plt_tools import (jackknife_mean_by_col,set_font_type, defaultPlotting)
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
from scipy.stats import ttest_rel, ttest_ind
from plot_functions.plt_tools import jackknife_list


# %%
##### Parameters to change #####
pick_data = 'a_gtau_box' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
if_jackknnife = False # Whether to jackknife (bool)

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'astro_BF1_{which_ztime}'
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
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_angVel_postBout', 'propBoutIEI_angVel_preNextBout', 'propBoutIEI_angVel','propBoutIEI_pauseDur','ztime','expNum','cond0','cond1','exp']]
IBI_angles_cond.columns = ['IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = IBI_angles_cond.ztime.unique()
all_ztime.sort()

sel_features = ['IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur']
# Calculate std:# %% tidy data
all_feature_cond = IBI_angles_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# %%
# assign up and down 

all_feature_toplt = all_feature_cond.copy()

# %%
# Plots

# %%
#mean
cat_cols = ['cond1','expNum','ztime','cond0','bout_time', 'exp']
feature_to_plt = [c for c in all_feature_toplt.columns if c not in cat_cols]
feature_for_comp = feature_to_plt + ['expNum']
# jackknife
all_feature_sampled = all_feature_toplt


cat_cols = ['cond0','cond1','ztime','expNum']
mean_data = all_feature_sampled.groupby(cat_cols).median()
mean_data = mean_data.reset_index()

cat_cols = ['cond0','cond1','ztime']

mean_data_jackknife = all_feature_sampled.groupby(cat_cols)[feature_for_comp].apply(
    lambda x: jackknife_mean_by_col(x,'expNum','median')
 )
mean_data_jackknife = mean_data_jackknife.reset_index()
# %%

####################################
###### Plotting Starts Here ######
####################################

if if_jackknnife:
    toplt = mean_data_jackknife
    x_name = 'cond1'
    gridrow = 'cond0'
    gridcol = 'ztime'
    units = 'jackknife_idx'
    prename = 'jackknifed__'
else: 
    toplt = mean_data
    x_name = 'cond1'
    gridrow = 'cond0'
    gridcol = 'ztime'
    units = 'expNum'
    prename = ''


feature_to_plt = ['IBI_pauseDur']

for feature in feature_to_plt:
    g = plt_categorical_grid2(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=True,
        height = 3,
        aspect = 1
        )
    filename = os.path.join(fig_dir,f"{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
    for ztime in ['day', 'night']:
        this_compare = toplt.query('ztime == @ztime')
        print(ztime)
        ttest_res, ttest_p = ttest_rel(this_compare.loc[this_compare['cond1']==all_cond1[0],feature],
                                this_compare.loc[this_compare['cond1']==all_cond1[1],feature])
        print(f'{feature} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')


# # %%
# for feature in feature_to_plt:
#     g = sns.catplot(
#         data=all_feature_toplt,
#         x='cond1',
#         col='ztime',
#         y=feature,
#         kind='point',
#     )
#     for ztime in ['day', 'night']:
#         this_compare = all_feature_toplt.query('ztime == @ztime')
#         print(ztime)
#         ttest_res, ttest_p = ttest_ind(this_compare.loc[this_compare['cond1']==all_cond1[0],feature],
#                                 this_compare.loc[this_compare['cond1']==all_cond1[1],feature])
#         print(f'{feature} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')
#     plt.savefig(fig_dir+f"/{feature} point.pdf",format='PDF')

# %%
sleep_threshold = 10

# all_feature_toplt_jackcol = all_feature_toplt.assign(jackknifecol=all_feature_toplt['cond1']+all_feature_toplt['ztime']+all_feature_toplt['expNum'].astype(str))
# all_feature_toplt_jackcol = all_feature_toplt_jackcol.assign(
#     expNum_jack = None
#     )
jackknife_res = pd.DataFrame()
for (cond1, ztime), all_feature_toplt_jackcol in all_feature_toplt.groupby(['cond1','ztime']):
    all_feature_toplt_jackcol = all_feature_toplt_jackcol.assign(expNum_jack = None)
    for i, list in enumerate(jackknife_list(all_feature_toplt_jackcol.expNum.unique())):
        thisSample = all_feature_toplt_jackcol.loc[all_feature_toplt_jackcol['expNum'].isin(list)].copy()
        thisSample['expNum_jack'] = i
        jackknife_res = pd.concat([jackknife_res,thisSample])
      
#%

if if_jackknnife:
    data_toplt = jackknife_res
    rep_col = 'expNum_jack'
else:
    data_toplt = all_feature_toplt
    rep_col = 'expNum'
left = data_toplt.query("IBI_pauseDur > @sleep_threshold").groupby(['cond1','ztime',rep_col]).size().reset_index()
right = data_toplt.groupby(['cond1','ztime',rep_col]).size().reset_index()
count = left.merge(right, on=['cond1','ztime',rep_col])
count.columns = ['cond1','ztime',rep_col, 'sleep', 'all']
count = count.assign(
    ratio_of_sleep = count['sleep']/count['all'] * 100
)
# %
feature = 'ratio_of_sleep'
g = plt_categorical_grid2(
    data=count,
    x_name='cond1',
    gridcol='ztime',
    gridrow=None,
    y_name=feature,
    units=rep_col,
    height=2
)
for ztime in ['day', 'night']:
    this_compare = count.query('ztime == @ztime')
    print(ztime)
    ttest_res, ttest_p = ttest_rel(this_compare.loc[this_compare['cond1']==all_cond1[0],feature],
                            this_compare.loc[this_compare['cond1']==all_cond1[1],feature])
    print(f'{feature} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')
plt.savefig(fig_dir+f"/{feature} of sleep point.pdf",format='PDF')

# %
# %%
