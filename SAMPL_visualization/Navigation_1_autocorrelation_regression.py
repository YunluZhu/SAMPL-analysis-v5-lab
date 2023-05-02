# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_consecutive_features import cal_autocorrelation_feature
from plot_functions.plt_functions import plt_categorical_grid
import scipy.stats as st
from plot_functions.plt_tools import (set_font_type)
from plot_functions.plt_tools import jackknife_list


##### Parameters to change #####

pick_data = 'hc' # name of your cond0 to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 8 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True

feature_AutoCorrelation = 'traj_peak'
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

if if_jackknife:
    folder_jackknife_ornot = 'jackknife'
else:
    folder_jackknife_ornot = 'byExpNum'
folder_name = f'Navi1_autoCorrReg_z{which_ztime}' + folder_jackknife_ornot
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %%
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE)


all_features = all_features.assign(
    epoch_uid = all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_uid = all_features['cond1'] + all_features['expNum'].astype(str),
)
    
# %%  autocorrelation-----------------
max_lag = consecutive_bout_num-1
col = 'expNum'
if_jackknife = True
autoCorr_res_jackknifed = pd.DataFrame()
jackknife_std = pd.DataFrame()

for (cond0, cond1), group in all_features.groupby(['cond0', 'cond1']):
    exp_df = group.groupby(col).size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]

    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group[col].isin(exp_group),:]
        this_corr_res, this_shifted_df, this_df_tocorr = cal_autocorrelation_feature(this_group_data, feature_AutoCorrelation, 'epoch_uid', max_lag)
        this_corr_res = this_corr_res.assign(
            cond1 = cond1,
            cond0 = cond0,
            expNum = j
        )
        autoCorr_res_jackknifed = pd.concat([autoCorr_res_jackknifed, this_corr_res], ignore_index=True)

autoCorr_res_jackknifed = autoCorr_res_jackknifed.reset_index()   
autoCorr_res_jackknifed = autoCorr_res_jackknifed.assign(
    r_sq = autoCorr_res_jackknifed[f'autocorr_{feature_AutoCorrelation}'] ** 2
)

# %%  Autocorrelation
all_cond1 = autoCorr_res_jackknifed.cond1.unique()

g = sns.relplot(
    data=autoCorr_res_jackknifed,
    x='lag',
    y='slope',
    hue='cond1',
    errorbar=('ci', 95),
    col='cond0',
    # row='cond1',
    kind='line',
    height=3
)
plt.savefig(fig_dir+f"/slope {feature_AutoCorrelation}.pdf",format='PDF')


g = sns.relplot(
    data=autoCorr_res_jackknifed,
    x='lag',
    y='r_sq',
    hue='cond1',
    errorbar=('ci', 95),
    col='cond0',
    # row='cond1',
    kind='line',
    height=3
)
plt.savefig(fig_dir+f"/R_square {feature_AutoCorrelation}.pdf",format='PDF')
