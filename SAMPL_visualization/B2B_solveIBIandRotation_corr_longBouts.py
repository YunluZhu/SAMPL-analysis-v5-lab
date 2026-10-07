'''


'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)
from plot_functions.get_bout_consecutive_features import extract_consecutive_bout_features
from plot_functions.plt_functions import plt_categorical_combined_3
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel
from lmfit import Model
import scipy.stats as st


set_font_type()

# %%
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' 'night', or 'all'
if_day_light_narrow_bin = False
##### Parameters to change #####

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

#%%

root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime, day_light_narrow_bin=if_day_light_narrow_bin)

# tidy bout uid
all_features = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)
    
all_features = all_features.loc[all_features['ztime'].isin(['day','night'])].reset_index(drop=True)
# %% std of directions of consecutive bouts
list_of_features = [
    'traj_peak',
    # 'post_IBI_time',
    'pre_IBI_time',
    
    'spd_peak',
    
    'pitch_end', 
    'pitch_initial',
    
    'rot_total',
    # 'rot_full_accel',
    # 'rot_l_decel',
    # 'rot_l_accel',
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts

#####################
max_lag = 1
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)
# take absolute value of x parameters
consecutive_bout_features = consecutive_bout_features.assign(
    # xdispl_swim = np.abs(consecutive_bout_features['xdispl_swim']),
    # x_pre_swim = np.abs(consecutive_bout_features['x_pre_swim']),
    # x_post_swim = np.abs(consecutive_bout_features['x_post_swim']),
    # x_initial = np.abs(consecutive_bout_features['x_initial']),
    # x_end = np.abs(consecutive_bout_features['x_end']),
)
# %% 
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond0', 'cond1','ztime','id']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    # swim to swim, 5 mm/s
    # post_S2S_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values, np.nan),
    # pre_S2S_ydispl = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values),
    # bout to bout, aligned, using next initial - previous end
    # post_B2B_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['y_end'].values, np.nan),
    # ydispl_bout = sel_consecutive_bouts.iloc[:,:]['y_end'].values - sel_consecutive_bouts.iloc[:,:]['y_initial'].values,
    
    # x displacement
    # post_S2S_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values, np.nan)),
    # pre_S2S_xdispl = np.abs(np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values)),
    # post_B2B_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['x_end'].values, np.nan)),
    # xdispl_bout = np.abs(sel_consecutive_bouts.iloc[:,:]['x_end'].values - sel_consecutive_bouts.iloc[:,:]['x_initial'].values),
    
    # rotation
    # post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),
    bouts = sel_consecutive_bouts['lag'] + 1,
)
#%%

# IMPORTANT: let's grab the second bout if there're only 2 consecutive bouts
# IMPORTANT: because did the above calculation in the messy way, we can only pick the middle bouts. Drop the first and last bout of each series
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)
middle_bout_df = middle_bout_df.loc[middle_bout_df['ztime'].isin(['day','night'])].reset_index(drop=True)

middle_bout_df = middle_bout_df.assign(
    traj_deviation = middle_bout_df['traj_peak']- middle_bout_df['pitch_initial'],
)
which_IBI = 'pre'

# IBI_threshold = middle_bout_df.groupby(['cond0','cond1'])[f'{which_IBI}_IBI_time'].transform(np.percentile, 50)
middle_bout_df['IBI_threshold'] = 1.7
middle_bout_df['IBI_cat'] = 'long_IBI'
middle_bout_df.loc[middle_bout_df[f'{which_IBI}_IBI_time'] < middle_bout_df['IBI_threshold'], 'IBI_cat'] = 'short_IBI'

middle_bout_df['traj_cat'] = pd.cut(middle_bout_df['traj_peak'], bins=[-np.inf,0, np.inf], labels=['dive','climb'])
middle_bout_df['initialPitch_cat'] = pd.cut(middle_bout_df['pitch_initial'], bins=[-np.inf,0, np.inf], labels=['initial_DN','initial_UP'])

#%%

df_toplt = middle_bout_df.groupby(['cond1','cond0']).head(5000)#.loc[middle_bout_df['IBI_cat']=='long_IBI']
df_toplt = df_toplt.query("cond0 == '07'")

xval='pre_B2B_rot'
yval='rot_total'

# scatter
sns.relplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',

    hue='cond1',
    palette=['black','black'],
    row='cond1',
    col='IBI_cat',
    kind='scatter', 
    height=2.5,
    alpha=0.01,
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.5, 99.5])
    }
    # cmap
)
plt.savefig(os.path.join(fig_dir, f"IBI rotation scatter {yval} {xval}.pdf"),format='PDF')

#%%
df_toplt = middle_bout_df.groupby(['cond1']).head(4000)#.loc[middle_bout_df['IBI_cat']=='long_IBI']
xval=f'pre_B2B_rot'
yval='rot_total'

# scatter
sns.lmplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',
    hue='cond1',
    palette=['black','black'],
    row='IBI_cat',
    col='cond1',
    height=2.5,
    # alpha=0.03,
    scatter_kws={'alpha':0.015},
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.5, 99.5])
    }
    # cmap
)
plt.savefig(os.path.join(fig_dir, f"IBI rotation lm {yval} {xval}.pdf"),format='PDF')


#%%
df_toplt = middle_bout_df.groupby(['cond1']).head(3000)#.loc[middle_bout_df['IBI_cat']=='long_IBI']
xval=f'pre_B2B_rot'
yval='traj_peak'

# scatter
sns.relplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',
    hue='cond1',
    palette=['black','black'],
    row='IBI_cat',
    col='cond1',
    height=2.5,
    # alpha=0.03,
    alpha=0.01,
    facet_kws={'sharey': 'row', 'sharex': 'row'}
)
plt.savefig(os.path.join(fig_dir, f"Exploratory lm {yval} {xval}.pdf"),format='PDF')

#%%
# linear regression for each condition and IBI category
regression_results = {}
for cond in middle_bout_df['cond1'].unique():
    for ibi_cat in middle_bout_df['IBI_cat'].unique():
        subset = middle_bout_df[(middle_bout_df['cond1'] == cond) & (middle_bout_df['IBI_cat'] == ibi_cat)]
        x = subset[xval].values
        y = subset[yval].values
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
        
        regression_results[(cond, ibi_cat)] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'r_squared': r_value**2
        }

# 

#%
#%%
# ratio of each IBI category calculate by expNum
IBI_ratio = middle_bout_df.groupby(['cond0','cond1','IBI_cat','expNum']).size().reset_index(name='count')
IBI_ratio = IBI_ratio.groupby(['cond0','cond1','expNum']).apply(lambda x: x.assign(ratio=x['count']/x['count'].sum()), include_groups=True).reset_index(drop=True)


plt_categorical_combined_3(
    data=IBI_ratio.query("IBI_cat=='long_IBI'"),
    x='cond0',
    col='cond1',
    y='ratio',
    units='expNum',
    errorbar='se',
    height=2.5,
)
plt.savefig(os.path.join(fig_dir, f"longIBI ratio by expNum.pdf"),format='PDF')

# run ttest
from scipy.stats import ttest_ind
longIBI_ratio = IBI_ratio.query("IBI_cat=='long_IBI'").groupby(['cond0'])['ratio'].apply(list)
t_stat, p_val = ttest_ind(longIBI_ratio[0], longIBI_ratio[1])
print(f"t-test result for longIBI ratio between control and experimental: t-statistic={t_stat}, p-value={p_val}")

# %%
# plot WHM distribution in all_feature_cond
sns.displot(
    data=all_feature_cond,
    x='WHM',
    stat='density',
    hue='cond1',
    kind='hist',
    bins=30,
    element='poly',
    height=3,
    common_norm=False,
)

# %%
sns.displot(
    data=all_feature_cond,
    x='pre_IBI_time',
    stat='density',
    hue='cond1',
    kind='hist',
    bins=30,
    element='poly',
    height=3,
    common_norm=False,
    log_scale=(True,False)
)
# %%
all_feature_cond['swim_frequency'] = 1/all_feature_cond['pre_IBI_time']
sns.displot(
    data=all_feature_cond,
    x='swim_frequency',
    stat='density',
    hue='cond1',
    kind='hist',
    bins=30,
    element='poly',
    height=2.5,
    common_norm=False,
    log_scale=True,
)
plt.savefig(os.path.join(fig_dir, f"swim_frequency distribution.pdf"),format='PDF')
# %%
