'''
plot mean IBI bout frequency vs. IBI pitch and fit with a parabola
UP DN separated

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - ONE sample number for day and night
- change the var RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change it to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_sum, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid
import random
##### Parameters to change #####
pick_data = 'a_rtau_box' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI7_freq'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')
set_font_type()
defaultPlotting()

# %%
IBI_angles, cond0_all, cond1_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

# %%
IBI_angles = IBI_angles.sort_values(by=['cond0','cond1','expNum']).reset_index(drop=True)

# %%
incre = 2
hour_bins = np.arange(0,24+incre,incre)

IBI_angles = IBI_angles.assign(
    hour = IBI_angles.propBoutIEItime.dt.hour+0.1
)
IBI_angles = IBI_angles.assign(
    binned_hour = pd.cut(IBI_angles['hour'], hour_bins)
)
IBI_angles = IBI_angles.assign(
    zt = IBI_angles.hour - 9
)
IBI_angles.loc[IBI_angles['zt']< 0, 'zt'] = IBI_angles.loc[IBI_angles['zt']< 0, 'zt'] + 24
IBI_angles = IBI_angles.assign(
    binned_zt = pd.cut(IBI_angles['zt'], hour_bins)
)

# %%

each_exp_cat_col = ['cond0', 'cond1', 'expNum']
feature_sel = 'bout_freq'
all_res = pd.DataFrame()
for (cond0, cond1, exp), group in IBI_angles.groupby(each_exp_cat_col):
    this_cond_res = distribution_binned_average(
        df=group,
        by_col='zt',
        bin_col=feature_sel,
        bin=hour_bins,
        method='median'
    )
    this_cond_res = this_cond_res.assign(
        cond0 = cond0,
        cond1 = cond1,
        expNum = exp
    )
    this_cond_res.index.names=['hour_bin']
    this_cond_res = this_cond_res.reset_index()
    all_res = pd.concat([all_res, this_cond_res])
all_res = all_res.assign(
    mid_hour = list(hour_bins[:-1]+incre/2) * len(IBI_angles.groupby(each_exp_cat_col).size())
)
# all_res_dup = all_res.copy()
# all_res_dup['mid_hour'] = all_res_dup['mid_hour']+24
# all_res = pd.concat([all_res,all_res_dup], ignore_index=True)

# %
sns.lineplot(
    data=all_res,
    hue='cond1',
    x='mid_hour',
    y=feature_sel,
    estimator=None,
    units='expNum',
    alpha=0.2
)
sns.lineplot(
    data=all_res,
    hue='cond1',
    x='mid_hour',
    y=feature_sel,
    errorbar=None
)
# %%
sns.relplot(
    kind='line',
    data=all_res,
    hue='cond1',
    x='mid_hour',
    y=feature_sel,
    row='cond0',
    errorbar='sd',
)
# %%

BOOT_REP = 20
IBI_angles_boot_ = []
for (cond0, cond1), group in IBI_angles.groupby(['cond0','cond1']):
    bst_input = group.index
    if BOOT_REP > 1:
        bst_output = [
            np.array(random.choices(bst_input, k=len(bst_input)))
            for i in np.arange(BOOT_REP)
        ]
        for bts, ind in enumerate(bst_output):
            bootdf = group.loc[ind, :]
            bootdf = bootdf.assign(
                bts_rep = bts
            )
            IBI_angles_boot_.append(bootdf)
IBI_angles_boot = pd.concat(IBI_angles_boot_, ignore_index=True)

#%%
each_exp_cat_col = ['cond0', 'cond1', 'bts_rep']
feature_sel = 'bout_freq'
all_res = pd.DataFrame()
for (cond0, cond1, exp), group in IBI_angles_boot.groupby(each_exp_cat_col):
    this_cond_res = distribution_binned_average(
        df=group,
        by_col='zt',
        bin_col=feature_sel,
        bin=hour_bins,
        method='median'
    )
    this_cond_res = this_cond_res.assign(
        cond0 = cond0,
        cond1 = cond1,
        rep = exp
    )
    this_cond_res.index.names=['hour_bin']
    this_cond_res = this_cond_res.reset_index()
    all_res = pd.concat([all_res, this_cond_res])
all_res = all_res.assign(
    mid_hour = list(hour_bins[:-1]+incre/2) * len(IBI_angles_boot.groupby(each_exp_cat_col).size())
)

#%%
# %
sns.relplot(
    kind='line',
    data=all_res,
    hue='cond1',
    x='mid_hour',
    y=feature_sel,
    row='cond0',
    errorbar='sd',
    height=3,
    aspect=1.6
)
plt.savefig(fig_dir+f"/time of day freq.pdf",format='PDF')

# %%
