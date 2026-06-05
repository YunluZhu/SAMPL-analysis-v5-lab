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
which_ztime = 'all' # 'day' 'night', or 'all'
if_jackknife = False # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'IBI3_avg_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI3_avg_z{which_ztime}'
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
IBI_angles_cond = IBI_angles#.loc[:,['propBoutIEI_angVel_postBout', 'propBoutIEI_angVel_preNextBout', 'propBoutIEI_angVel','propBoutIEI_pauseDur','ztime','expNum','cond0','cond1','exp']]
# IBI_angles_cond.columns = ['IBI_angvel_postBout','IBI_angvel_preBout', 'IBI_angvel', 'IBI_pauseDur','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1','expNum','boutNum','exp']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

sel_features_bool = [ele not in cond_cols for ele in IBI_angles_cond.columns]
sel_features = IBI_angles_cond.columns[sel_features_bool]



# %%
# angaccel
df_toplt = df.query("propBoutIEI_strict_angAcc > -60 and propBoutIEI_strict_angAcc < 20")
g = sns.histplot(data=df_toplt, x='propBoutIEI_strict_angAcc', hue='cond1', common_norm=False, )
# g.set(xlim=[-.75,.30])



for (condition, ztime), group in df.groupby(['cond1','ztime']):
    print(condition, ztime)
    print(group['propBoutIEI_strict_angAcc'].median())

#%%
IBI_min_threshold = 0.3

g = sns.displot(
    data=IBI_angles_cond.query("propBoutIEI_angAcc > -70 and propBoutIEI_angAcc < 30 and propBoutIEI_pauseDur >= @IBI_min_threshold"),
    x='propBoutIEI_pauseDur',
    y='propBoutIEI_angAcc',
    col='cond1',
    row='ztime',
    common_norm=False
    # alpha=0.1
)
g.set(ylim=[-30,20],
      xlim=[0,15]
      )


#%%
IBI_threshold = 2.5

for (condition, ztime), group in df.groupby(['cond1','ztime']):
    print(condition, ztime)
    
    print(group.query("propBoutIEI_pauseDur >= @IBI_threshold")[['propBoutIEI_strict_angAcc','propBoutIEI_strict_yAcce']].mean())

# %%
# angaccel
df_toplt = IBI_angles_cond.query("propBoutIEI_pauseDur >= @IBI_threshold")
df_toplt = df_toplt.query("propBoutIEI_angAcc > -60 and propBoutIEI_angAcc < 20")
g = sns.histplot(data=df_toplt, x='propBoutIEI_angAcc', hue='cond1', common_norm=False, )
g.set(xlim=[np.percentile(df_toplt['propBoutIEI_angAcc'],1),np.percentile(df_toplt['propBoutIEI_angAcc'],99)])

# %%




#%%
df_toplt = df
x='propBoutIEI_pauseDur'
y='propBoutIEI_strict_angAcc'
g = sns.displot(
    data=df_toplt.query("propBoutIEI_pauseDur >= 1.2"),
    x=x,
    y=y,
    col='cond1',
    row='ztime',
    common_norm=False
    # alpha=0.1
)
g.set(
    xlim=[np.percentile(df_toplt[x].dropna(),1),np.percentile(df_toplt[x].dropna(),99)],
    # ylim=[np.percentile(df_toplt[y].dropna(),5),np.percentile(df_toplt[y].dropna(),97)],
    ylim=[-10,5]
    )

# %%
df_toplt = df
x='propBoutIEI_strict_pitch'
y='propBoutIEI_strict_spd'
g = sns.displot(
    data=df_toplt.query("propBoutIEI_pauseDur >= 0"),
    x=x,
    y=y,
    col='cond1',
    row='ztime',
    common_norm=False
    # alpha=0.1
)
g.set(
    xlim=[np.percentile(df_toplt[x].dropna(),1),np.percentile(df_toplt[x].dropna(),99)],
    ylim=[np.percentile(df_toplt[y].dropna(),3),np.percentile(df_toplt[y].dropna(),97)],
    # ylim=[-10,5]
    )