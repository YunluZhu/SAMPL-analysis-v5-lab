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
# from astropy.stats import jackknife_resampling
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_avg2, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid
from scipy.stats import ttest_rel

##### Parameters to change #####
pick_data = 'a_gtau'
which_ztime = 'all' # 'day' 'night', or 'all'
if_jackknife = False # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'sleep_z{which_ztime}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'sleep_z{which_ztime}'
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
IBI_angles_cond = IBI_angles
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()

# Calculate std:
# std_jackknife v= IBI_std_cond.groupby(cond_cols).apply(
    #     lambda x: jackknife_mean(x)
    # )
    # std_jackknife = std_jackknife.loc[:,'IBI_pitch'].reset_index()
    
# %%
x_name = 'propBoutIEI'

# x_range = np.arange(-35,40,0.5)

df_toplt_filtered = IBI_angles_cond.dropna(subset=[x_name])
# df_toplt_filtered = df_toplt_filtered.query("pre_IBI_time > 4")
df_toplt_filtered = df_toplt_filtered.sort_values(by=cond_cols)
binned_angles_raw = pd.DataFrame()

g = sns.displot(
    df_toplt_filtered,
    x=x_name,
    col='cond0',
    row='ztime',
    height=2.5,
    hue='cond1',
    stat='density',
    element = 'poly',
    log_scale=True,
    bins='scott',
    common_norm=False,
)
g.set(
    xlim=[np.percentile(df_toplt_filtered[x_name],0.2), np.percentile(df_toplt_filtered[x_name],99.8)],
)   

plt.savefig(fig_dir+f"/{x_name} dist.pdf",format='PDF')

# %%
x_name='cond1'
y_name='propBoutIEI'
units='expNum'
gridcol='ztime'

toplt_df = df_toplt_filtered.groupby([x_name, gridcol,units],group_keys=False)[y_name].median().reset_index()
plt_categorical_grid(
    data=toplt_df,
    x_name=x_name,
    y_name=y_name,
    units=units,
    gridcol=gridcol,
    gridrow=None,
    height=2.5,

)

#%%
groupSizeN = df_toplt_filtered.groupby([x_name, gridcol, units],group_keys=False).size()
groupSize = int(groupSizeN.median())

repeated_p = np.ndarray(shape=(1,len(df_toplt_filtered[gridcol].unique())))
for rep in np.arange(100):
    all_cat_p = []
    if if_jackknife:
        toplt_df = jackknife_avg2(df_toplt_filtered, [x_name, gridcol], units, y_name, method='median')
    else:
        tocalc_df = df_toplt_filtered.groupby([x_name, gridcol,units],group_keys=False).sample(groupSize,replace=True).reset_index()
        toplt_df = tocalc_df.groupby([x_name, gridcol,units],group_keys=False)[y_name].median().reset_index()
    for _, df in toplt_df.groupby(gridcol):
        ttest_res, ttest_p = ttest_rel(df.loc[df[x_name]==cond1[0],y_name],
                                    df.loc[df[x_name]==cond1[1],y_name])
        all_cat_p.append(ttest_p)
    repeated_p = np.append(repeated_p, [all_cat_p], axis=0)
repeated_p = repeated_p[1:,:]
print(repeated_p.mean(axis=0))

# %%

# %%

# %%
