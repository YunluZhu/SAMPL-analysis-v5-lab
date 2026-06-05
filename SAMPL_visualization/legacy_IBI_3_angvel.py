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
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid2

##### Parameters to change #####
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' 'night', or 'all'
if_jackknife = False # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'IBI3_std_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI3_std_z{which_ztime}'
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

IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_angVel_postBout', 'propBoutIEI_angVel_preNextBout', 'propBoutIEI_angVel','propBoutIEI_pauseDur','ztime','expNum','cond0','cond1','exp']]
IBI_angles_cond.columns = ['IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

sel_features = ['IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur']
# Calculate std:
# std_jackknife v= IBI_std_cond.groupby(cond_cols).apply(
    #     lambda x: jackknife_mean(x)
    # )
    # std_jackknife = std_jackknife.loc[:,to_plot_feature].reset_index()

#%%

#%%
sns.displot(
    kind='kde',
    data=IBI_angles_cond,
    x='IBI_angvel',
    y='IBI_pauseDur',
    col='cond1',
    common_norm=False
)
# %%
sns.displot(
    kind='kde',
    data=IBI_angles_cond,
    x='IBI_angvel_preBout',
    y='IBI_pauseDur',
    col='cond1',
    common_norm=False
)
# %%
# %%
sns.displot(
    kind='kde',
    data=IBI_angles_cond,
    x='IBI_angvel_postBout',
    y='IBI_pauseDur',
    col='cond1',
    common_norm=False
)
# %%
