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
consecutive_bout_num = 6 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True

list_of_features_StandardDeviation = ['traj_peak']
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

if if_jackknife:
    folder_jackknife_ornot = 'jackknife'
else:
    folder_jackknife_ornot = 'byExpNum'
folder_name = f'Navi3_featureSD_z{which_ztime}' + folder_jackknife_ornot
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
    
# %% jackknife std of consecutive bout
# 

jackknife_col = 'expNum'
max_lag = consecutive_bout_num-1

jackknife_std = pd.DataFrame()
for (cond0, cond1), group in all_features.groupby(['cond0', 'cond1']):
    exp_df = group.groupby(jackknife_col).size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]

    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group[jackknife_col].isin(exp_group),:]
        this_jackknife_std = this_group_data.std(numeric_only=True).to_frame().T
        jackknife_std = pd.concat([jackknife_std,
                                    this_jackknife_std.assign(
                                        cond0 = cond0,
                                        cond1 = cond1,
                                        jackknife_group = j
                                    )],ignore_index=True)


for feature_toplt in list_of_features_StandardDeviation:
    plt.figure(figsize=(4,4))
    plt_categorical_grid(
        data=jackknife_std,
        x_name='cond1',
        y_name=feature_toplt,
        units='jackknife_group',
        gridcol='cond0',
        gridrow=None,
        errorbar='sd',
        markertype='o'
    )
    plt.savefig(fig_dir+f"/std_allBouts {feature_toplt}.pdf",format='PDF')

    # print(feature_toplt)
    # x = df_toplt.loc[df_toplt['cond1'] == df_toplt.cond1.unique()[0], feature_toplt]
    # y = df_toplt.loc[df_toplt['cond1'] == df_toplt.cond1.unique()[1], feature_toplt]
    # print(f'*{thisds}')
    # print(st.ttest_rel(x, y))
