# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.plt_functions import plt_categorical_grid
import scipy.stats as st
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from plot_functions.plt_tools import jackknife_list


pick_data = 'wt_lightR' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF3_distPolar_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)

# %% tidy data
col_to_adj = 'pitch_peak' 
# origin for x is on the right. To make it North0East, use preX-postX to correct posture calculation 
all_feature_cond['N0E_' + col_to_adj] = (all_feature_cond.x_pre_swim - all_feature_cond.x_post_swim)/(all_feature_cond.x_post_swim - all_feature_cond.x_pre_swim).abs() * (90 + all_feature_cond[col_to_adj]) + 180
# col_to_adj = 'propBoutIEI_pitch'
# all_ibi_cond = all_ibi_cond['N0E_' + col_to_adj] = (all_ibi_cond.x_post_swim - all_ibi_cond.x_pre_swim)/(all_ibi_cond.x_post_swim - all_ibi_cond.x_pre_swim).abs() * (90 + all_ibi_cond[col_to_adj]) + 180
    
# %% 
col_toplt = 'N0E_' + col_to_adj
# col_toplt = col_to_adj
df_toplt = all_feature_cond#.query("cond0 == @all_cond0[1]")

min_val = 0
max_val = 360
step = (max_val-min_val)/50
bins = np.arange(min_val,max_val+step,step)

angle_counts = df_toplt.groupby(['cond1']).apply(
    lambda g: np.histogram(g[col_toplt], bins)[0]/len(g)
)

bin_mid = (bins[1:] + bins[:-1])/2
# %
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
for i, cond in enumerate(df_toplt.cond1.unique()):
    ax.plot(np.radians(bin_mid), angle_counts[i])

plt.savefig(os.path.join(fig_dir, f"bout direction hist.pdf"),format='PDF')


# %%
