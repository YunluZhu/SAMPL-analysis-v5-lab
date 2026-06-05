'''
Selected a list of features
Plot standard deviation of each feature during series of bouts given by consecutive_bout_num

NOTE variables to keep an eye on:

pick_data # name of your cond0 to plot as defined in function get_data_dir()
which_ztime # 'day', 'night', or 'all'
consecutive_bout_num = 6 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True # or False
list_of_features_StandardDeviation = ['traj_peak']  # select a list of features here

'''

# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_consecutive_features import cal_autocorrelation_feature
from plot_functions.get_bout_consecutive_features import (extract_consecutive_bout_features)

from plot_functions.plt_functions import plt_categorical_grid2
# import scipy.stats as st
from plot_functions.plt_tools import (set_font_type)
from plot_functions.plt_tools import jackknife_list


##### Parameters to change #####

pick_data = 'creTau8' # name of your cond0 to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 5 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = False

list_of_features_StandardDeviation = ['traj_peak', 'pitch_peak']  # select a list of features here
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

if if_jackknife:
    folder_jackknife_ornot = 'jackknife'
else:
    folder_jackknife_ornot = 'byExpNum'
folder_name = f'Navi3_conseqRMS_z{which_ztime}' + folder_jackknife_ornot
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %%
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime)


all_features = all_features.assign(
    epoch_conduid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_conduid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str),
)
    
# %% jackknife std of consecutive bout
# 

max_lag = consecutive_bout_num-1

consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features_StandardDeviation, max_lag)

#%%

for feature in list_of_features_StandardDeviation:
    consecutive_bout_features[f'{feature}_RMS'] = (consecutive_bout_features[feature] - consecutive_bout_features[f'{feature}_first'])**2

feature_toplt = [feature + '_RMS' for feature in list_of_features_StandardDeviation]
#%%
jackknife_col = 'expNum'

jackknife_res = pd.DataFrame()
for (cond0, cond1), group in consecutive_bout_features.groupby(['cond0', 'cond1']):
    exp_df = group.groupby(jackknife_col).size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]

    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group[jackknife_col].isin(exp_group),:]
        this_jackknife_res = this_group_data.groupby('id', group_keys=False)[feature_toplt].apply(
            lambda x: (np.sum(x)/(len(x)-1))**0.5
            ).reset_index()
        jackknife_res = pd.concat([jackknife_res,
                                    this_jackknife_res.assign(
                                        cond0 = cond0,
                                        cond1 = cond1,
                                        jackknife_group = j
                                    )],ignore_index=True)

#%%
for ft in feature_toplt:
    plt.figure(figsize=(4,4))
    plt_categorical_grid2(
        data=jackknife_res,
        x_name='cond1',
        y_name=ft,
        units='jackknife_group',
        gridcol='cond0',
        gridrow=None,
        errorbar='sd',
        markertype='o'
    )
    plt.savefig(fig_dir+f"/rms_consecutiveBouts {ft}.pdf",format='PDF')

    # print(feature_toplt)
    # x = df_toplt.loc[df_toplt['cond1'] == df_toplt.cond1.unique()[0], feature_toplt]
    # y = df_toplt.loc[df_toplt['cond1'] == df_toplt.cond1.unique()[1], feature_toplt]
    # print(f'*{thisds}')
    # print(st.ttest_rel(x, y))

# %%
