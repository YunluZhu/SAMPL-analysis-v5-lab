'''

'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean, set_font_type, defaultPlotting, distribution_binned_average)

set_font_type()
# defaultPlotting()
# %%
pick_data = 'wt_fin'
which_zeitgeber = 'all' # Day only!!!!!!!
DAY_RESAMPLE = 1000
NIGHT_RESAMPLE = 500

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-30,40,0.1)
BIN_WIDTH = 1
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK5_steering_phased_z{which_zeitgeber}_sample{DAY_RESAMPLE}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond0, all_cond0 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)

# %% fit sigmoid - master
all_coef = pd.DataFrame()
all_y = pd.DataFrame()
# all_binned_average = pd.DataFrame()

def func(x,k,b):
    y = k * x + b
    return y

# df_tofit = all_feature_cond.loc[all_feature_cond['spd_peak']>5,:]
# df_tofit = all_feature_cond
# if DAY_RESAMPLE != 0:
#     df_tofit = all_feature_cond.groupby(
#             ['cond0','cond1','expNum']
#             ).sample(
#                     n=DAY_RESAMPLE,
#                     replace=True
#                     )

all_feature_day = pd.DataFrame()
if which_zeitgeber != 'night':
    all_feature_day = all_feature_cond.loc[
        all_feature_cond['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        all_feature_day = all_feature_day.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )

all_feature_night = pd.DataFrame()
if which_zeitgeber != 'day':
    all_feature_night = all_feature_cond.loc[
        all_feature_cond['ztime']=='night',:
            ]
    if NIGHT_RESAMPLE != 0:  # if resampled
        all_feature_night = all_feature_night.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True
                        )


df_tofit = pd.concat([all_feature_day,all_feature_night],ignore_index=True)

for (cond1,cond1,this_ztime), for_fit in df_tofit.groupby(['cond1','cond0','ztime']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_for_fit = for_fit.loc[for_fit['expNum'].isin(idx_group)]
        k, y_intersect = np.polyfit(x = this_for_fit['pitch_peak_phase'], 
                          y = this_for_fit['traj_peak'],
                          deg = 1) 
        x_intersect = -1 * y_intersect / k
        all_coef = pd.concat([all_coef, pd.DataFrame(
            data = {
                'k': k,
                'y_intersect': y_intersect,
                'x_intersect': x_intersect,
                'cond0': cond1,
                'cond1': cond1,
                'excluded_exp': excluded_exp,
                'ztime': this_ztime,
            }, index=[0]
        )
                              ], ignore_index=True)
        
        y = func(X_RANGE,k, y_intersect)
        this_y = pd.DataFrame(data=y, columns=['y']).assign(
            x=X_RANGE,
            cond0 = cond1,
            cond1 = cond1,
            excluded_exp = excluded_exp,
            ztime = this_ztime,)
        all_y = pd.concat([all_y, this_y], ignore_index=True)

all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()
# %%
# plot fitted line
plt.figure()

g = sns.relplot(x='x',y='y', data=all_y, 
                kind='line',
                col='cond0',
                row = 'ztime', 
                hue='cond1',
                ci='sd',
                )
g.set(xlim=(-30, 40))
g.set(ylim=(-30, 40))

filename = os.path.join(fig_dir,"steering fit.pdf")
plt.savefig(filename,format='PDF')
# %%
# plot coefs
plt.close()
    
for feature in ['k',	'y_intersect',	'x_intersect']:
    p = sns.catplot(
        data = all_coef, y=feature,x='cond1',kind='point',join=False,
        col='cond0',
        ci='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='cond1', dodge=True,
        hue_order = all_cond0,
        aspect=0.6, sharey='row'
    )
    p.map(sns.lineplot,'cond1',feature,
          estimator=None,
        units='excluded_exp',
        # hue='cond1',
        color='grey',
        alpha=0.2,
        data=all_coef)
    filename = os.path.join(fig_dir,f"coef by age {feature}.pdf")
    plt.savefig(filename,format='PDF')

# %%
plt.close()
    
for feature in ['k',	'y_intersect',	'x_intersect']:
    p = sns.catplot(
        data = all_coef, y=feature,x='ztime',kind='point',join=False,
        col='cond1',
        ci='sd',
        row = 'cond0', 
        # units=excluded_exp,
        hue='cond1', dodge=True,
        hue_order = all_cond0,
        aspect=0.6
    )
    p.map(sns.lineplot,'ztime',feature,
          estimator=None,
        units='excluded_exp',
        # hue='cond1',
        color='grey',
        alpha=0.2,
        data=all_coef)
    filename = os.path.join(fig_dir,f"coef by ztime {feature}.pdf")
    plt.savefig(filename,format='PDF')

# %%
