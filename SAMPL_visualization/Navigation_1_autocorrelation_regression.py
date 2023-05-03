'''
Selected a feature
Plot feature values of bout N+1 vs bout N in scatter, and fit with line
Plot Aurocorrelation coefficient of determination (R2) of bout(0) vs bout(0+lag)
Plot slope of bout(0) vs bout(0+lag)

NOTE variables to keep an eye on:

pick_data # name of your cond0 to plot as defined in function get_data_dir()
which_ztime # 'day', 'night', or 'all'
consecutive_bout_num = 8 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True # or False
feature_AutoCorrelation = 'traj_peak'  # select a feature here

'''

# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)

import scipy.stats as st
from plot_functions.plt_tools import (set_font_type)
from plot_functions.plt_tools import jackknife_list


##### Parameters to change #####

pick_data = 'hc' # name of your cond0 to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 8 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True # True or False

feature_AutoCorrelation = 'traj_peak' # select a feature here
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

if if_jackknife:
    folder_jackknife_ornot = 'jackknife'
else:
    folder_jackknife_ornot = 'byExpNum'
folder_name = f'Navi1_autoCorrReg_z{which_ztime}' + folder_jackknife_ornot
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %%
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE)


all_features = all_features.assign(
    epoch_conduid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_conduid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str),
)
    
# %% get feature bout n+1 vs bout n
max_lag = 1
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, [feature_AutoCorrelation], max_lag)
df_toplt = consecutive_bout_features.query("lag == 1").sort_values(by=['id']).reset_index(drop=True)
df_toplt.columns = [f'{feature_AutoCorrelation}_N' if x==f'{feature_AutoCorrelation}_first' else x for x in df_toplt.columns]
df_toplt.columns = [f'{feature_AutoCorrelation}_N+1' if x==f'{feature_AutoCorrelation}' else x for x in df_toplt.columns]

# %% plot scatter

xmin = np.percentile(df_toplt[f'{feature_AutoCorrelation}_N'].values,0.5)
xmax = np.percentile(df_toplt[f'{feature_AutoCorrelation}_N'].values,99.5)

g = sns.FacetGrid(
    data=df_toplt, 
    # x=f'{feature_AutoCorrelation}_N', 
    # y=f'{feature_AutoCorrelation}_N+1', 
    col='cond1',
    row='cond0',
    # kind="scatter", 
    # alpha=0.05, 
    aspect = 1, 
    height = 3,
    # linewidths=0
    ylim = (xmin, xmax),
    xlim = (xmin, xmax),
    )

for (row_val, col_val), ax in g.axes_dict.items():
    this_cond_data = df_toplt.query("cond0 == @row_val & cond1 == @col_val")
    xval = this_cond_data[f'{feature_AutoCorrelation}_N'].values
    yval = this_cond_data[f'{feature_AutoCorrelation}_N+1'].values
    this_slope, this_intercept, this_r, this_p, this_se = st.linregress(xval, yval)
    X_plot = np.linspace(xmin, xmax, 1000)
    Y_plot = this_slope * X_plot + this_intercept
    if len(this_cond_data) > 4000:
        this_cond_data = this_cond_data.sample(4000) 
    sns.scatterplot(
        data=this_cond_data, 
        x=f'{feature_AutoCorrelation}_N', 
        y=f'{feature_AutoCorrelation}_N+1', 
        alpha=0.05,
        linewidths=0,
        ax=ax
    )
    sns.lineplot(x=X_plot, y=Y_plot, color='r', ax=ax)
    ax.text(0, xmin+8,f'slope = {this_slope:.3f}', fontsize=9) #add text
    ax.text(0, xmin+2,f'r = {this_r:.3f}', fontsize=9) #add text

plt.savefig(fig_dir+f"/{feature_AutoCorrelation} N+1 vs N scatter.pdf",format='PDF')
# print(f"{feature_toplt} Pearson correlation coeff: {this_r}")

# %%  autocorrelation-----------------
max_lag = consecutive_bout_num-1
col = 'expNum'
if_jackknife = True
autoCorr_res_jackknifed = pd.DataFrame()
jackknife_std = pd.DataFrame()

for (cond0, cond1), group in all_features.groupby(['cond0', 'cond1']):
    exp_df = group.groupby(col).size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]

    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group[col].isin(exp_group),:]
        this_corr_res, _, _ = cal_autocorrelation_feature(this_group_data, feature_AutoCorrelation, 'epoch_conduid', max_lag)
        this_corr_res = this_corr_res.assign(
            cond1 = cond1,
            cond0 = cond0,
            expNum = j
        )
        autoCorr_res_jackknifed = pd.concat([autoCorr_res_jackknifed, this_corr_res], ignore_index=True)

autoCorr_res_jackknifed = autoCorr_res_jackknifed.reset_index()   
autoCorr_res_jackknifed = autoCorr_res_jackknifed.assign(
    r_sq = autoCorr_res_jackknifed[f'autocorr_{feature_AutoCorrelation}'] ** 2
)

# %%  Autocorrelation
all_cond1 = autoCorr_res_jackknifed.cond1.unique()

g = sns.relplot(
    data=autoCorr_res_jackknifed,
    x='lag',
    y='slope',
    hue='cond1',
    errorbar=('ci', 95),
    col='cond0',
    # row='cond1',
    kind='line',
    height=3
)
plt.savefig(fig_dir+f"/slope {feature_AutoCorrelation}.pdf",format='PDF')


g = sns.relplot(
    data=autoCorr_res_jackknifed,
    x='lag',
    y='r_sq',
    hue='cond1',
    errorbar=('ci', 95),
    col='cond0',
    # row='cond1',
    kind='line',
    height=3
)
plt.savefig(fig_dir+f"/R_square {feature_AutoCorrelation}.pdf",format='PDF')

# %%
