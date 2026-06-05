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
from astropy.stats import jackknife_resampling
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid2

##### Parameters to change #####
pick_data = 'a_rtau_box' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
if_jackknife = False # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'IBI5_IBIdur_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'IBI5_IBIdur_z{which_ztime}'
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
IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','propBoutIEI_pauseDur','ztime','expNum','cond0','cond1','exp']]
IBI_angles_cond.columns = ['IBI_pitch', 'IBI_pauseDur','ztime','expNum','cond0','cond1','exp']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()
IBI_angles_cond = IBI_angles_cond.assign(
    freq = 1/IBI_angles_cond['IBI_pauseDur']
)

# %%

####################################
###### Plotting Starts Here ######
####################################

# # plot kde of all
# g = sns.FacetGrid(IBI_angles_cond, 
#                 row="ztime", row_order=all_ztime,
#                 col='cond0', col_order=cond0,
#                 hue='cond1', hue_order=cond1,
#                 )
# g.map(sns.kdeplot, to_plot_feature,alpha=0.5,)
# g.add_legend()
# filename = os.path.join(fig_dir,f"{to_plot_feature} kde.pdf")
# plt.savefig(filename,format='PDF')
IBI_angles_cond = IBI_angles_cond.dropna()
x_name = 'IBI_pitch'
y_name = 'freq'
gridrow = 'cond0'
gridcol = 'cond1'
units = 'expNum'
g = sns.relplot(
    data = IBI_angles_cond,
    x = x_name,
    y = y_name,
    row = gridrow,
    col = gridcol,
    height = 3,
    aspect = 1,
    alpha=0.01,
    )
g.set(
    ylim=[np.percentile(IBI_angles_cond[y_name].values,2),
          np.percentile(IBI_angles_cond[y_name].values,95)],
    xlim=[np.percentile(IBI_angles_cond[x_name].values,0.1),
          np.percentile(IBI_angles_cond[x_name].values,99.9)]
)
# filename = os.path.join(fig_dir,f"{prename}{to_plot_feature}STD__by{x_name}__{gridcol}X{gridrow}.pdf")
# plt.savefig(filename,format='PDF')
# plt.show()

# %%

# %%

# %% check IBI duration, avoid having fish sticking on the wall
g = sns.displot(
    data=IBI_angles_cond,
    x='IBI_pauseDur',
    hue='cond1',
    col='cond0',
    kind='ecdf',
    # bins='scott',
    log_scale=True,
    # common_norm=False,
)
g.set(
    xlim=[np.percentile(IBI_angles_cond.IBI_pauseDur, 0), np.percentile(IBI_angles_cond.IBI_pauseDur, 99.5)]
)
# %%

# %%
