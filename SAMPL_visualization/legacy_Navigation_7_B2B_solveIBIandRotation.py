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
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)
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
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
if_day_light_narrow_bin = True
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'Navi7_B2B_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass

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
    # 'bout_trajectory_Pre2Post',
    # 'y_post_swim','y_pre_swim', 'ydispl_swim', 
    # 'y_end','y_initial',
    
    # 'post_IBI_align_time',
    'post_IBI_time',
    # 'pre_IBI_align_time',
    'pre_IBI_time',
    
    'spd_peak',
    
    'pitch_end', 
    'pitch_initial',
    'pitch_peak',
    'pitch_post_bout', # where righting ends
    'rot_total',
    'rot_full_accel',
    'rot_l_decel',
    'rot_l_accel',
    # grab x parameters
    # 'x_pre_swim','x_post_swim',
    # 'x_end','x_initial',
    # 'xdispl_swim',
    # 'atk_ang'
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts

#####################
max_lag = 2
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
    post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),

    bouts = sel_consecutive_bouts['lag'] + 1,
)



# #%% Two activity modes
# # split into two df based on lag
# first_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==0].reset_index(drop=True)
# second_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)

# # calculate traj peak change
# first_bout_df = first_bout_df.assign(
#     traj_change_abs = np.abs(second_bout_df['traj_peak']- first_bout_df['traj_peak']),
#     traj_change = second_bout_df['traj_peak']- first_bout_df['traj_peak'],
#     spd_change = second_bout_df['spd_peak']- first_bout_df['spd_peak'],
# )

# binned_avg = pd.DataFrame()

# bins = np.percentile(first_bout_df['post_IBI_time'], np.arange(0.5, 100, 8))
# for (cond0, cond1, expNum), group in first_bout_df.groupby(['cond0','cond1', 'expNum']):
#     # moving average by post IBI time
#     this_binned_avg = distribution_binned_average_opt(
#         df=group,
#         by_col='post_IBI_time',
#         bin_col='traj_change_abs',
#         bin=bins,
#         method='median',
#     )
#     this_binned_avg = this_binned_avg.assign(
#         cond0 = cond0,
#         cond1 = cond1,
#         expNum = expNum,
#         bin_mean = bins[:-1] + np.diff(bins) / 2,
#     )
#     binned_avg = pd.concat([binned_avg, this_binned_avg], ignore_index=True)

# #%%
# g = sns.relplot(
#     kind='line',
#     data=binned_avg,
#     x='bin_mean',
#     y='traj_change_abs',
#     col='cond0',
#     row='cond1',
#     height=3,
# )
# # loop through column and row of plot g

# # Loop through rows and columns
# for row_num, row_axes in enumerate(g.axes):
#     for col_num, ax in enumerate(row_axes):
#         # Get the data for the current row and column
#         cond0 = g.col_names[col_num]
#         cond1 = g.row_names[row_num]
        
#         # Filter the DataFrame for the current row and column
#         group = first_bout_df[(first_bout_df['cond0'] == cond0) & (first_bout_df['cond1'] == cond1)]
        
#         # # log transform of post_IBI_time
#         # log_transformed_time = np.log10(group['post_IBI_time'])
        
#         # # fit with normal distribution to the post_IBI_time
#         # mu, std = st.norm.fit(log_transformed_time)
        
#         # Calculate the median of the x-axis values
#         median_x = group['post_IBI_time'].median()
        
        
#         # Plot the vertical line at the median x value
#         ax.axvline(median_x, color='red', linestyle='--')
#         ax.set_title(f'{cond0} {cond1}')


# # %% Consistency and IBI duration
# # generate a dataframe with pivot by lag, only take traj_peak
# first_two_bouts = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']<2].reset_index(drop=True)
# pivot_df = first_two_bouts.pivot_table(
#     index=['cond0','cond1','expNum','id'],
#     columns=['lag'],
#     values=['traj_peak','post_IBI_time'],
#     aggfunc='first',
# ).reset_index(drop=False)
# # flattern multiindex
# pivot_df.columns = [f'{f}_{s}' if s != '' else f'{f}' for f, s in pivot_df.columns]

# # split into 4 quadrants based on post_IBI_time_0
# pivot_df['IBI_cat'] = pd.cut(
#     pivot_df['post_IBI_time_0'],
#     bins=pivot_df['post_IBI_time_0'].quantile([0, 0.25, 0.5, 0.75, 1]),
#     labels=['IBI_Q1', 'IBI_Q2', 'IBI_Q3', 'IBI_Q4'],
# )

# #%% plot traj_peak vs traj_peak
# sns.displot(
#     bins=200,
#     data=pivot_df,
#     x='traj_peak_0',
#     y='traj_peak_1',
#     col='IBI_cat',
#     row='cond0',
#     common_norm=False,
#     height=3,
#     facet_kws={
#         'xlim': np.percentile(pivot_df['traj_peak_0'].dropna(),[.5, 99.5]),
#         'ylim': np.percentile(pivot_df['traj_peak_1'].dropna(),[.5, 99.5])
#     }
# )

#%% 

#########
#########
#########


# IMPORTANT: because did the above calculation in the messy way, we can only pick the middle bouts. Drop the first and last bout of each series
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)
middle_bout_df = middle_bout_df.loc[middle_bout_df['ztime'].isin(['day','night'])].reset_index(drop=True)

middle_bout_df = middle_bout_df.assign(
    traj_deviation = middle_bout_df['traj_peak']- middle_bout_df['pitch_initial'],
)
which_IBI = 'pre'

middle_bout_df['IBI_threshold'] = 1.7#middle_bout_df.groupby(['cond0','cond1'])[f'{which_IBI}_IBI_time'].transform(np.percentile, 50)
middle_bout_df['IBI_cat'] = 'long_IBI'
middle_bout_df.loc[middle_bout_df[f'{which_IBI}_IBI_time'] < middle_bout_df['IBI_threshold'], 'IBI_cat'] = 'short_IBI'

middle_bout_df['traj_cat'] = pd.cut(middle_bout_df['traj_peak'], bins=[-np.inf,0, np.inf], labels=['dive','climb'])
middle_bout_df['initialPitch_cat'] = pd.cut(middle_bout_df['pitch_initial'], bins=[-np.inf,5, np.inf], labels=['initial_DN','initial_UP'])

middle_bout_df['time_cat'] = pd.cut(
    middle_bout_df['pre_IBI_time'],
    bins=np.append([0], (middle_bout_df.query("IBI_cat == 'long_IBI'")['pre_IBI_time'].quantile([0, 0.25, 0.5, 0.75, 1]))),
    labels=['short','LongIBI_Q1', 'LongIBI_Q2', 'LongIBI_Q3', 'LongIBI_Q4'],
)
middle_bout_df['time_cat2'] = pd.cut(
    middle_bout_df['pre_IBI_time'],
    bins=np.append([0], (middle_bout_df.query("IBI_cat == 'long_IBI'")['pre_IBI_time'].quantile([0, 0.5, 1]))),
    labels=['short','Long1', 'Long2'],
)

# %%

xval=f'{which_IBI}_B2B_rot'
yval='rot_total'
sns.relplot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    row='cond1',
    hue='traj_peak',
    # use diverge color pallette
    palette=sns.diverging_palette(240, 10, as_cmap=True),
    height=3,
    alpha=0.05,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"scatter {yval} {xval}.pdf"),format='PDF')



xval=f'{which_IBI}_B2B_rot'
yval='rot_l_accel'
sns.relplot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    row='cond1',
    hue='traj_peak',
    # use diverge color pallette
    palette=sns.diverging_palette(240, 10, as_cmap=True),
    height=3,
    alpha=0.05,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"scatter {yval} {xval}.pdf"),format='PDF')

#%%
# plot again in kde

xval=f'{which_IBI}_B2B_rot'
yval='rot_total'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    row='cond1',
    common_norm=False,
    height=3,
    pthresh=0.1,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval}.pdf"),format='PDF')

#%%

# does trajectory matter?
which_cat = 'traj_cat'

xval=f'{which_IBI}_B2B_rot'
yval='rot_total'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    pthresh=0.1,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval} by {which_cat}.pdf"),format='PDF')

# trajectory does not matter

#     %%
# does posture initial matter?

which_cat = 'initialPitch_cat'

xval=f'{which_IBI}_B2B_rot'
yval='rot_total'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    levels=8,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval} by {which_cat}.pdf"),format='PDF')

#%% check if these bouts with long IBI still follows righting reflex

which_cat = None

xval=f'pitch_initial'
yval='rot_l_decel'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    levels=8,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval} by {which_cat}.pdf"),format='PDF')


#%%
# color code righting scatter plot by IBI time
xval=f'pitch_initial'
yval='rot_l_decel'

g = sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='time_cat',
    row='cond1',
    hue='time_cat',
    height=3,
    levels=8,
    common_norm=False,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.1, 99.9]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.1, 99.9])
    }
)
plt.savefig(os.path.join(fig_dir, f"kde {yval} {xval} by log IBI time.pdf"),format='PDF')
#%%
# plot it in another way, distribution binned average

# 2 D point plot with error bars on x and y
xval=f'pitch_initial'
yval='rot_l_decel'
bins = np.percentile(middle_bout_df[xval], np.arange(2, 100, 10))
binned_avg = pd.DataFrame()
for (cond0, cond1, tmcat, expNum), group in middle_bout_df.groupby(['cond0','cond1', 'time_cat2','expNum']):
    # moving average by post IBI time
    this_binned_avg = distribution_binned_average_opt(
        df=group,
        by_col=xval,
        bin_col=yval,
        bin=bins,
        method='median',
    )
    this_binned_avg = this_binned_avg.assign(
        cond0 = cond0,
        cond1 = cond1,
        time_cat = tmcat,
        expNum = expNum,
        bin_mean = [0.5*(x.left + x.right)  for x in this_binned_avg.index],
    )
    binned_avg = pd.concat([binned_avg, this_binned_avg], ignore_index=True)

#%%
binned_avg = binned_avg.assign(
    IBI_cat = ['short' if x == 'short' else 'long_IBI' for x in binned_avg['time_cat'].values],
)
g = sns.relplot(
    kind='line',
    data=binned_avg,
    x='bin_mean',
    y=yval,
    # col='IBI_cat',
    row='cond1',
    hue='time_cat',
    height=3, 
    # errorbar=None,
)
plt.savefig(os.path.join(fig_dir, f"Righting binned {yval} {xval} by IBI time.pdf"),format='PDF')


#%% 
# # try by IBI rotation
# middle_bout_df['ibi_rot_cat'] = pd.cut(
#     middle_bout_df['pre_B2B_rot'],
#     bins= middle_bout_df['pre_B2B_rot'].quantile([0, .25, 0.5, .75, 1]),
#     labels=['IBI_rot_Q1', 'IBI_rot_Q2', 'IBI_rot_Q3', 'IBI_rot_Q4'],
# )

# set ibi_rot_cat based on pre_B2B_rot and quantiles by long and short IBI
long_df = middle_bout_df.copy()#.query("IBI_cat == 'long_IBI'")

long_df['ibi_rot_cat'] = pd.cut(
    long_df['pre_B2B_rot'],
    # bins= long_df['pre_B2B_rot'].quantile([0, 0.33,.66, 1]),
    bins= [-45, -15, -5, -1],
    labels=['longIBI_45-15', 'longIBI_15-5', 'longIBI_5-1'],
)

# 2 D point plot with error bars on x and y
xval=f'pitch_initial'
yval='rot_l_decel'
# bins = np.percentile(long_df[xval], np.arange(2, 100, 16))
bins = [-25, -10, -5, 0, 5, 10, 15]
binned_avg = pd.DataFrame()
for (cond0, cond1, tmcat, expNum), group in long_df.groupby(['cond0','cond1', 'ibi_rot_cat','expNum']):
    # moving average by post IBI time
    this_binned_avg = distribution_binned_average_opt(
        df=group,
        by_col=xval,
        bin_col=yval,
        bin=bins,
        method='median',
    )
    this_binned_avg = this_binned_avg.assign(
        cond0 = cond0,
        cond1 = cond1,
        time_cat = tmcat,
        expNum = expNum,
        bin_mean = [0.5*(x.left + x.right)  for x in this_binned_avg.index],
    )
    binned_avg = pd.concat([binned_avg, this_binned_avg], ignore_index=True)

g = sns.relplot(
    kind='line',
    data=binned_avg,
    x='bin_mean',
    y=yval,
    # col='IBI_cat',
    row='cond1',
    hue='time_cat',
    height=3, 
    # errorbar=None,
)
plt.savefig(os.path.join(fig_dir, f"Righting binned {yval} {xval} by IBI rot.pdf"),format='PDF')


#%%
# direct measurement of effective righting rotation
############################################
#%%

import scipy.optimize as opt  # noqa: E402

def bisquare_weights(residuals, c=4.685):
        """Calculates bisquare weights for given residuals.

        Args:
            residuals (numpy.ndarray): Array of residuals.
            c (float, optional): Tuning constant. Defaults to 4.685.

        Returns:
            numpy.ndarray: Array of bisquare weights.
        """
        abs_residuals = np.abs(residuals)
        weights = np.where(abs_residuals <= c, (1 - (residuals / c)**2)**2, 0)
        return weights
    
def weighted_linear_fit(x, y, weights):
    """Performs weighted linear regression.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.
        weights (numpy.ndarray): Weights for each data point.

    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    def objective_function(params):
        slope, intercept = params
        y_predicted = slope * x + intercept
        return np.sum(weights * (y - y_predicted)**2)

    initial_guess = [0, 0]  # Initial guess for slope and intercept
    result = opt.minimize(objective_function, initial_guess)
    slope, intercept = result.x
    return slope, intercept

def bisquare_linear_fit(x, y, max_iter=100, tolerance=1e-6):
    """Performs bisquare weighted linear fit iteratively.

    Args:
        x (numpy.ndarray): Independent variable data.
        y (numpy.ndarray): Dependent variable data.
        max_iter (int, optional): Maximum iterations. Defaults to 100.
        tolerance (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
            tuple: Slope and intercept of the fitted line.
    """
    weights = np.ones_like(y)  # Initial weights are all 1
    for _ in range(max_iter):
        prev_slope, prev_intercept = weighted_linear_fit(x, y, weights)
        residuals = y - (prev_slope * x + prev_intercept)
        new_weights = bisquare_weights(residuals)

        # Check for convergence
        if np.allclose(new_weights, weights, rtol=tolerance):
            break
        weights = new_weights

    return prev_slope, prev_intercept


middleBout_setPoint = middle_bout_df.groupby(
    ['cond0','cond1']
)[['pitch_initial','rot_l_decel']].apply(
    lambda g: pd.Series(bisquare_linear_fit(x=g['pitch_initial'], y=g['rot_l_decel']))
)
middleBout_setPoint.columns=['righting_slope','righting_intercept']

middle_bout_df_SteeringRighting = middle_bout_df.merge(
    middleBout_setPoint.reset_index(),
    on=['cond0','cond1'],
)

#%
middleBout_st = middle_bout_df.groupby(
    ['cond0','cond1']
)[['traj_deviation','rot_full_accel']].apply(
    lambda g: pd.Series(bisquare_linear_fit(x=g['traj_deviation'], y=g['rot_full_accel']))
)
middleBout_st.columns=['steering_slope','steering_intercept']
#% Calculate steering
middle_bout_df_SteeringRighting = middle_bout_df_SteeringRighting.merge(
    middleBout_st.reset_index(),
    on=['cond0','cond1'],
)

#%%
middle_bout_df_SteeringRighting = middle_bout_df_SteeringRighting.assign(
    theoretical_steering = middle_bout_df_SteeringRighting['traj_deviation'] * middle_bout_df_SteeringRighting['steering_slope'] + middle_bout_df_SteeringRighting['steering_intercept'],
    theoretical_righting = middle_bout_df_SteeringRighting['pitch_initial'] * middle_bout_df_SteeringRighting['righting_slope'] + middle_bout_df_SteeringRighting['righting_intercept'],
    rot_post2end = middle_bout_df_SteeringRighting['pitch_end'] - middle_bout_df_SteeringRighting['pitch_post_bout'],
)

middle_bout_df_SteeringRighting = middle_bout_df_SteeringRighting.assign(
    theoretical_rotTotal = middle_bout_df_SteeringRighting['theoretical_righting'] + middle_bout_df_SteeringRighting['theoretical_steering'] + middle_bout_df_SteeringRighting['rot_post2end'],
)
middle_bout_df_SteeringRighting = middle_bout_df_SteeringRighting.assign(
    excessive_rot = middle_bout_df_SteeringRighting['rot_total'] - middle_bout_df_SteeringRighting['theoretical_rotTotal'],
)
#%%
sns.displot(
    kind='hist',
    stat='probability',
    element='poly',
    common_norm=False,
    data=middle_bout_df_SteeringRighting,
    hue='cond1',
    col='IBI_cat',
    x='excessive_rot'
)
plt.savefig(os.path.join(fig_dir, f"excessive rot full bout.pdf"),format='PDF')

#%%

xval=f'{which_IBI}_B2B_rot'
yval='excessive_rot'

sns.displot(
    kind='kde',
    data=middle_bout_df_SteeringRighting,
    x=xval,
    y=yval,
    col='IBI_cat',
    row='cond1',
    common_norm=False,
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df_SteeringRighting[xval].dropna(),[.1, 99.9]),
        'ylim': np.percentile(middle_bout_df_SteeringRighting[yval].dropna(),[.1, 99.9])
    }
)
plt.savefig(os.path.join(fig_dir, f"middle_bout_df_SteeringRighting kde {yval} {xval}.pdf"),format='PDF')

#%%
# color code by speed

xval=f'{which_IBI}_B2B_rot'
yval='excessive_rot'

sns.relplot(
    kind='scatter',
    data=middle_bout_df_SteeringRighting.loc[middle_bout_df_SteeringRighting['spd_peak']<30],
    x=xval,
    y=yval,
    col='IBI_cat',
    row='cond1',
    hue='spd_peak',
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df_SteeringRighting[xval].dropna(),[.1, 99.9]),
        'ylim': np.percentile(middle_bout_df_SteeringRighting[yval].dropna(),[.1, 99.9])
    },
    alpha=0.02,
    palette='viridis',
)
plt.savefig(os.path.join(fig_dir, f"middle_bout_df_SteeringRighting scatter {yval} {xval} by spd.pdf"),format='PDF')





#%%

xval=f'{which_IBI}_B2B_rot'
yval='spd_peak'

sns.displot(
    kind='kde',
    data=middle_bout_df_SteeringRighting.loc[middle_bout_df_SteeringRighting['spd_peak']<30],
    x=xval,
    y=yval,
    row='cond1',
    common_norm=False,
    hue='IBI_cat',
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df_SteeringRighting[xval].dropna(),[.3, 99.7]),
        'ylim': np.percentile(middle_bout_df_SteeringRighting[yval].dropna(),[.1, 99.9])
    },
)
plt.savefig(os.path.join(fig_dir, f"middle_bout_df_SteeringRighting kde {yval} {xval} .pdf"),format='PDF')

#%%






#%% check if these bouts with long IBI still follows steering

which_cat = None

xval=f'traj_deviation'
yval='rot_full_accel'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    levels=8,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval} by {which_cat}.pdf"),format='PDF')

which_cat = None

xval=f'pre_B2B_rot'
yval='traj_deviation'

sns.displot(
    kind='kde',
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='IBI_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    levels=8,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.2, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"hist {yval} {xval} by {which_cat}.pdf"),format='PDF')
