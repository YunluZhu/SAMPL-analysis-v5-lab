'''

'''

#%%
import os
import pandas as pd # pandas library
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
pick_data = '7dd_all'
which_zeitgeber = 'day' # Day only!!!!!!!
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-30,40,0.1)
BIN_WIDTH = 1
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK5_righting_test_{which_zeitgeber}_sample{DAY_RESAMPLE}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)

# %% calculate some angles

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
#             ['dpf','condition','expNum']
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
                ['dpf','condition','expNum']
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
                ['dpf','condition','expNum']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True
                        )

# %%
all_y = pd.DataFrame()
all_coef = pd.DataFrame()
xfeature = 'pitch_pre_bout'
yfeature = 'rot_l_decel'

df_tofit = pd.concat([all_feature_day,all_feature_night],ignore_index=True)

for (cond1,cond2,this_ztime), for_fit in df_tofit.groupby(['condition','dpf','ztime']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_for_fit = for_fit.loc[for_fit['expNum'].isin(idx_group)]
        k, y_intersect = np.polyfit(x = this_for_fit[xfeature], 
                          y = this_for_fit[yfeature],
                          deg = 1) 
        x_intersect = -1 * y_intersect / k
        all_coef = pd.concat([all_coef, pd.DataFrame(
            data = {
                'k': k,
                'y_intersect': y_intersect,
                'x_intersect': x_intersect,
                'dpf': cond2,
                'condition': cond1,
                'excluded_exp': excluded_exp,
                'ztime': this_ztime,
            }, index=[0]
        )
                              ], ignore_index=True)
        
        y = func(X_RANGE,k, y_intersect)
        this_y = pd.DataFrame(data=y, columns=['y']).assign(
            x=X_RANGE,
            dpf = cond2,
            condition = cond1,
            excluded_exp = excluded_exp,
            ztime = this_ztime,)
        all_y = pd.concat([all_y, this_y], ignore_index=True)

all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()

# plot fitted line
plt.figure()

g = sns.relplot(x='x',y='y', data=all_y, 
                kind='line',
                col='dpf',
                row = 'ztime', 
                hue='condition',
                ci='sd',
                )
g.map(sns.scatterplot,data = df_tofit.sample(5000),
      x = xfeature,
      y = yfeature,
      hue = 'condition',
      hue_order =  all_cond2,
      alpha = 0.1,
      )
# g.set(xlim=(-30, 40))
# g.set(ylim=(-4, 8))

filename = os.path.join(fig_dir,"righting fit.pdf")
plt.savefig(filename,format='PDF')
print(all_coef.groupby('condition')['k'].mean())
# %%
toplt = df_tofit
x = 'pitch_initial'
y = 'pitch_peak'
upper = np.percentile(toplt[x], 99)
lower = np.percentile(toplt[x], 1)
BIN_WIDTH = 10
AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
binned_df1 = toplt.groupby(['condition','dpf']).apply(
    lambda group: distribution_binned_average(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
)
binned_df1.columns=[x,y]

# repeat for pitch post bout
x = 'pitch_initial'
y = 'pitch_post_bout'
upper = np.percentile(toplt[x], 99)
lower = np.percentile(toplt[x], 1)
BIN_WIDTH = 10
AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
binned_df2 = toplt.groupby(['condition','dpf']).apply(
    lambda group: distribution_binned_average(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
)
binned_df2.columns=[x,y]

binned_df = binned_df1.assign(
    pitch_post_bout = binned_df2[y]
)

binned_df = binned_df.reset_index(level=['condition','dpf'])
binned_df = binned_df.reset_index(drop=True)
# %%
df_tofit_sample = binned_df
pitch_initial_df = df_tofit_sample.reset_index()[['index','pitch_initial']].rename(columns={'index':'bout_num','pitch_initial':'pitch'}).assign(
    what_pitch = 'pitch_initial',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_peak_df = df_tofit_sample.reset_index()[['index','pitch_peak']].rename(columns={'index':'bout_num','pitch_peak':'pitch'}).assign(
    what_pitch = 'pitch_peak',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_post_bout_df = df_tofit_sample.reset_index()[['index','pitch_post_bout']].rename(columns={'index':'bout_num','pitch_post_bout':'pitch'}).assign(
    what_pitch = 'pitch_post_bout',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_chg_df = pd.concat([pitch_initial_df, pitch_peak_df, pitch_post_bout_df],ignore_index=True)
pitch_chg_df = pitch_chg_df.sort_values(by=['bout_num','what_pitch'])

# plt.figure(figsize=(2,3))
g = sns.relplot(x='what_pitch',y='pitch', 
                data=pitch_chg_df, 
                kind='line',
                estimator=None,
                units='bout_num',
                # alpha=0.01,
                col='dpf',
                row = 'condition', 
                aspect=1,
                color='black',
                markers=True,
                # ci='sd',
                )
filename = os.path.join(fig_dir,"posture change 2.pdf")
plt.savefig(filename,format='PDF')





# %%
df_tofit_sample = df_tofit.sample(n=5000)
pitch_initial_df = df_tofit_sample.reset_index()[['index','pitch_initial']].rename(columns={'index':'bout_num','pitch_initial':'pitch'}).assign(
    what_pitch = 'pitch_initial',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_post_bout_df = df_tofit_sample.reset_index()[['index','pitch_post_bout']].rename(columns={'index':'bout_num','pitch_post_bout':'pitch'}).assign(
    what_pitch = 'pitch_post_bout',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_chg_df = pd.concat([pitch_initial_df, pitch_post_bout_df],ignore_index=True)
pitch_chg_df = pitch_chg_df.sort_values(by=['bout_num','what_pitch'])

# plt.figure(figsize=(2,3))
g = sns.relplot(x='what_pitch',y='pitch', 
                data=pitch_chg_df, 
                kind='line',
                estimator=None,
                units='bout_num',
                alpha=0.005,
                col='dpf',
                row = 'condition', 
                aspect=0.8,
                # ci='sd',
                )
filename = os.path.join(fig_dir,"posture change.pdf")
plt.savefig(filename,format='PDF')
# %%

# %%
df_tofit_sample = df_tofit.sample(n=5000)
pitch_peak_df = df_tofit_sample.reset_index()[['index','pitch_peak']].rename(columns={'index':'bout_num','pitch_peak':'pitch'}).assign(
    what_pitch = 'pitch_peak',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_post_bout_df = df_tofit_sample.reset_index()[['index','pitch_post_bout']].rename(columns={'index':'bout_num','pitch_post_bout':'pitch'}).assign(
    what_pitch = 'pitch_post_bout',
    condition = df_tofit_sample['condition'].values,
    dpf=df_tofit_sample['dpf'].values
)
pitch_chg_df = pd.concat([pitch_peak_df, pitch_post_bout_df],ignore_index=True)
pitch_chg_df = pitch_chg_df.sort_values(by=['bout_num','what_pitch'])

# plt.figure(figsize=(2,3))
g = sns.relplot(x='what_pitch',y='pitch', 
                data=pitch_chg_df, 
                kind='line',
                estimator=None,
                units='bout_num',
                alpha=0.005,
                col='dpf',
                row = 'condition', 
                aspect=0.8,
                # ci='sd',
                )
filename = os.path.join(fig_dir,"posture change.pdf")
plt.savefig(filename,format='PDF')