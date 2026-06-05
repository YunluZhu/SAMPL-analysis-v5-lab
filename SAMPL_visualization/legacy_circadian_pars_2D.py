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
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_functions import plt_categorical_grid
import random

#%%
##### Parameters to change #####
pick_data = 'sldp2025' # name of your dataset to plot as defined in function get_data_dir()
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
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond["dataset"] = pick_data
# %%
all_feature_cond = all_feature_cond.sort_values(by=['cond0','cond1','expNum']).reset_index(drop=True)

# %%
# incre = 2
# hour_bins = np.arange(0,24+incre,incre)

all_feature_cond = all_feature_cond.assign(
    hour = all_feature_cond.bout_time.dt.hour+0.1,
    date = all_feature_cond.bout_time.dt.date
)

all_feature_cond['date'] = pd.to_datetime(all_feature_cond['date'])
all_feature_cond['days_diff'] = all_feature_cond.groupby(['cond0', 'cond1', 'expNum'])['date'].transform(lambda x: x.diff()).dt.days.fillna(0)


all_feature_cond = all_feature_cond.assign(
    cum = all_feature_cond.groupby(['cond0','cond1','expNum'])['days_diff'].transform(lambda x: np.cumsum(x))
)

all_feature_cond = all_feature_cond.assign(
    dayHour = all_feature_cond.hour + all_feature_cond.cum * 24
)

all_feature_cond = all_feature_cond.assign(
    zt = all_feature_cond.dayHour - 9
)

incre = 2
hour_bins = np.arange(all_feature_cond['dayHour'].min()-.1,all_feature_cond['dayHour'].max()+incre,incre)
zt_bins = np.arange(all_feature_cond['zt'].min()-.1,all_feature_cond['zt'].max()+incre,incre)


all_feature_cond = all_feature_cond.assign(
    binned_hour = pd.cut(all_feature_cond['dayHour'], hour_bins),
    binned_zt = pd.cut(all_feature_cond['zt'], zt_bins)
)
# %%
pick_par = 'traj_peak'
for by_which, bin in zip(['dayHour', 'zt'], [hour_bins, zt_bins]):
    each_exp_cat_col = ['cond0', 'cond1', 'expNum']
    all_res = pd.DataFrame()
    for (cond0, cond1, exp), group in all_feature_cond.groupby(each_exp_cat_col):
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
        mid_hour = all_res.binned.apply(lambda x:x.mid)#list(bin[:-1]+incre/2) * len(all_feature_cond.groupby(each_exp_cat_col).size())
    )

# # %
# sns.lineplot(
#     data=all_res,
#     hue='cond1',
#     x='mid_hour',
#     y=feature_sel,
#     estimator=None,
#     units='expNum',
#     alpha=0.2
# )
# sns.lineplot(
#     data=all_res,
#     hue='cond1',
#     x='mid_hour',
#     y=feature_sel,
#     errorbar=None
# )
# # %%
    sns.relplot(
        kind='line',
        data=all_res,
        hue='cond1',
        x='mid_hour',
        y=pick_par,
        row='cond0',
        errorbar='sd',
    )
    plt.savefig(fig_dir+f"/time of day {pick_par} {by_which}.pdf",format='PDF')

# %%

BOOT_REP = 20
all_feature_cond_boot_ = []
for (cond0, cond1), group in all_feature_cond.groupby(['cond0','cond1']):
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
            all_feature_cond_boot_.append(bootdf)
all_feature_cond_boot = pd.concat(all_feature_cond_boot_, ignore_index=True)

tm = 9
for by_which, bin in zip(['dayHour', 'zt'], [hour_bins, zt_bins]):
    each_exp_cat_col = ['cond0', 'cond1', 'bts_rep']
    all_res = pd.DataFrame()
    for (cond0, cond1, exp), group in all_feature_cond_boot.groupby(each_exp_cat_col):
        this_cond_res = distribution_binned_average(
            df=group,
            by_col=by_which,
            bin_col='bout_freq',
            bin=bin,
            method='median'
        )
        this_cond_res = this_cond_res.assign(
            cond0 = cond0,
            cond1 = cond1,
            bts_rep = exp
        )
        this_cond_res.index.names=['binned']
        this_cond_res = this_cond_res.reset_index()
        all_res = pd.concat([all_res, this_cond_res], ignore_index=True)
    all_res = all_res.assign(
        mid_hour = list(bin[:-1]+incre/2) * len(all_feature_cond_boot.groupby(each_exp_cat_col).size())
    )

    g = sns.relplot(
        kind='line',
        data=all_res,
        hue='cond1',
        x='mid_hour',
        y='bout_freq',
        row='cond0',
        errorbar='sd',
        height=3
    )
    g.set(xlim=[tm,tm+48])
    plt.savefig(fig_dir+f"/time of day freq boot {by_which}.pdf",format='PDF')
    
    tm-=9

# %%
