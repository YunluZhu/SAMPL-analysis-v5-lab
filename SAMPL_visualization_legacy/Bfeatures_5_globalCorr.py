'''
Plots clustered heatmap of correlations among bout features
mostly useless
'''

#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_index import get_index

from plot_functions.plt_tools import (jackknife_mean,set_font_type, day_night_split, defaultPlotting)
import matplotlib as mpl

# %%
##### Parameters to change #####

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%

root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'BF_corr_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)
# %% get bout features and IBI data
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztim=which_ztime)

# %% tidy feature data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# get kinetics separated by dpf
all_kinetics = all_feature_cond.groupby(['cond0','cond1']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
ctrl_kinetics = all_kinetics.loc[all_kinetics['cond1']==all_cond1[0],:]

# assign up and down
all_feature_UD = pd.DataFrame()
all_feature_cond = all_feature_cond.assign(direction=np.nan)
for key, group in all_feature_cond.groupby(['cond0']):
    this_setvalue = ctrl_kinetics.loc[ctrl_kinetics['cond0']==key,'set_point'].to_list()[0]
    group['direction'] = pd.cut(group['pitch_initial'],
                                bins=[-91,this_setvalue,91],
                                labels=['dn','up'])
    all_feature_UD = pd.concat([all_feature_UD,group])

# %%
# calculate percentage of change, separated by age, repeats, direction and of course, conditions
cat_cols = ['bout_time', 'expNum', 'ztime', 'cond0', 'cond1', 'direction']
features = list(set(all_feature_UD.columns).difference(set(cat_cols)))

feature_chg_idx = pd.DataFrame()
for (this_direction, this_dpf, this_ztime), group in all_feature_UD.groupby(['direction','cond0','ztime']):
    this_ctrl = group.loc[group['cond1']==all_cond1[0],features]
    this_cond = group.loc[group['cond1']==all_cond1[1],features] # assuming 2 conditions
    this_ctrl_mean = this_ctrl.mean()
    this_ctrl_std = this_ctrl.std()
    this_cond_mean = this_cond.mean()
    
    this_cond_chg = ((this_cond_mean-this_ctrl_mean)/(this_ctrl_mean)).to_frame().T
    this_cond_chg = this_cond_chg.assign(
        direction = this_direction,
        cond0 = this_dpf,
        ztime = this_ztime,
        cond1 = all_cond1[1]
    )
    feature_chg_idx = pd.concat([feature_chg_idx, this_cond_chg],ignore_index=True)

# %% 
####################################
###### Plotting Starts Here ######
####################################
# Correlation plot and feature clustering of control data

for key, group in all_feature_UD.groupby(['cond0']):
    print(f"cond0 = {key}")
    df_to_corr = group.loc[group['cond1']==all_cond1[0],features].sort_index(axis = 1)
    corr = df_to_corr.corr()

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.clustermap(corr, 
                cmap=cmap, vmax=1,vmin=-1, center=0,
                linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(fig_dir+f"/ctrl feature corr {key}.pdf",format='PDF')
