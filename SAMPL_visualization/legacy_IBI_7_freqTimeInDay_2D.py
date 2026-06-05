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

from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_sum, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid
import random
##### Parameters to change #####
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI7_freq_2day'
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
# incre = 2
# hour_bins = np.arange(0,24+incre,incre)

IBI_angles = IBI_angles.assign(
    hour = IBI_angles.propBoutIEItime.dt.hour+0.1,
    date = IBI_angles.propBoutIEItime.dt.date
)

IBI_angles['date'] = pd.to_datetime(IBI_angles['date'])
IBI_angles['days_diff'] = IBI_angles.groupby(['cond0', 'cond1', 'expNum'])['date'].transform(lambda x: x.diff()).dt.days.fillna(0)


IBI_angles = IBI_angles.assign(
    cum = IBI_angles.groupby(['cond0','cond1','expNum'])['days_diff'].transform(lambda x: np.cumsum(x))
)

IBI_angles = IBI_angles.assign(
    dayHour = IBI_angles.hour + IBI_angles.cum * 24
)

IBI_angles = IBI_angles.assign(
    zt = IBI_angles.dayHour - 9
)

incre = 2
hour_bins = np.arange(IBI_angles['dayHour'].min()-.1,IBI_angles['dayHour'].max()+incre,incre)
zt_bins = np.arange(IBI_angles['zt'].min()-.1,IBI_angles['zt'].max()+incre,incre)


IBI_angles = IBI_angles.assign(
    binned_hour = pd.cut(IBI_angles['dayHour'], hour_bins),
    binned_zt = pd.cut(IBI_angles['zt'], zt_bins)
)
# %%

pick_par = 'bout_freq'
for by_which, bin in zip(['dayHour', 'zt'], [hour_bins, zt_bins]):
    each_exp_cat_col = ['cond0', 'cond1', 'expNum']
    all_res = pd.DataFrame()
    for (cond0, cond1, exp), group in IBI_angles.groupby(each_exp_cat_col):
        this_cond_res = distribution_binned_average(
            df=group,
            by_col=by_which,
            bin_col=pick_par,
            bin=bin,
            method='mad'
        )
        this_cond_res = this_cond_res.assign(
            cond0 = cond0,
            cond1 = cond1,
            expNum = exp
        )
        this_cond_res.index.names=['binned']
        this_cond_res = this_cond_res.reset_index()
        all_res = pd.concat([all_res, this_cond_res], ignore_index=True)
    all_res = all_res.assign(
        mid_hour = all_res.binned.apply(lambda x:x.mid)
    )

    sns.relplot(
        kind='line',
        data=all_res,
        hue='cond1',
        x='mid_hour',
        y=pick_par,
        row='cond0',
        errorbar='sd',
        aspect=1.8,
        height=2
    )
    plt.savefig(fig_dir+f"/time of day {pick_par} {by_which}.pdf",format='PDF')

# %%
