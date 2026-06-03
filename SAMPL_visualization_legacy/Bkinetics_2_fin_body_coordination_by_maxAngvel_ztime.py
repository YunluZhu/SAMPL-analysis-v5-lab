'''
plot fin-body ratio with rotation calculated using max adjusted angvel from each condition

Fin-body ratio with new definitions slightly different from eLife 2019. Works well.
plot attack angle vs. early body change (-250 to -40 ms, or to time of max angular velocity), fit with a sigmoid w/ 4-free parameters
'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_max_angvel_rot, get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid

##### Parameters to change #####
pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknife = False
if_use_maxAngvelTime_perCond1 = False # if to calculate max adjusted angvel time for each condition and selectt range for body rotation differently
                                        # or to use -250ms to -40ms for all conditions
##### Parameters to change #####

# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,0,-100,1]
    upper_bounds = [10,20,2,100]
    x0=[5, 1, 0, 5]
    
    for key, value in kwargs.items():
        if key == 'a':
            x0[0] = value
            lower_bounds[0] = value-0.01
            upper_bounds[0] = value+0.01
        elif key == 'b':
            x0[1] = value
            lower_bounds[1] = value-0.01
            upper_bounds[1] = value+0.01
        elif key == 'c':
            x0[2] = value
            lower_bounds[2] = value-0.01
            upper_bounds[2] = value+0.01
        elif key =='d':
            x0[3] = value
            lower_bounds[3] = value-0.01
            upper_bounds[3] = value+0.01
            
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df[which_rotation], df[which_atk_ang], 
                        #    maxfev=10000, 
                           p0 = p0,
                           bounds=(lower_bounds,upper_bounds))
    y = func(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-5,20.05,0.05)
BIN_WIDTH = 0.5
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK2_fin_body_maxAngvel_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
defaultPlotting(size=16)
# %% get max_angvel_time per condition
which_rotation = 'rot_to_max_angvel'
which_atk_ang = 'atk_ang' 

# get features
if if_use_maxAngvelTime_perCond1:
    max_angvel_time, all_cond0, all_cond1 = get_max_angvel_rot(root, FRAME_RATE, ztime = which_ztime)
    all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime = which_ztime, max_angvel_time = max_angvel_time)
else:
    all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime = which_ztime )

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
if FRAME_RATE > 100:
    df_toplt = all_feature_cond.drop(all_feature_cond.loc[(all_feature_cond['atk_ang']<0) & (all_feature_cond['rot_full_accel']>all_feature_cond['rot_full_accel'].median())].index)
    df_toplt = df_toplt.loc[df_toplt['spd_peak']>=7]
elif FRAME_RATE == 40:
    df_toplt = all_feature_cond.drop(all_feature_cond.loc[(all_feature_cond['atk_ang']<0) & (all_feature_cond['rot_full_accel']>all_feature_cond['rot_full_accel'].median())].index)
    df_toplt.drop(df_toplt[df_toplt['spd_peak']<4].index, inplace=True)

# %%
angles_day_resampled = pd.DataFrame()
angles_night_resampled = pd.DataFrame()

if which_ztime != 'night':
    angles_day_resampled = df_toplt.loc[
        df_toplt['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        angles_day_resampled = angles_day_resampled.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True,
                        # random_state=2
                        )
if which_ztime != 'day':
    angles_night_resampled = df_toplt.loc[
        df_toplt['ztime']=='night',:
            ]
    if NIGHT_RESAMPLE != 0:  # if resampled
        angles_night_resampled = angles_night_resampled.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True,
                        # random_state=2
                        )
df_toplt = pd.concat([angles_day_resampled,angles_night_resampled],ignore_index=True)

# %%
# scatter plot of raw data
toplt = df_toplt.assign(
    traj_dir = pd.cut(df_toplt['traj_peak'], bins=[-90,0,90],labels=['negTraj','posTraj'])
)
x,y = 'rot_to_max_angvel','atk_ang'
upper = np.percentile(toplt[x], 99)
lower = np.percentile(toplt[x], 2)
BIN_WIDTH = 0.5
AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
binned_df = toplt.groupby(['cond1','cond0']).apply(
    lambda group: distribution_binned_average(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
)
binned_df.columns=[x,y]
binned_df = binned_df.reset_index(level=['cond0','cond1'])
binned_df = binned_df.reset_index(drop=True)

# xlabel = "Relative pitch change (deg)"
# ylabel = 'Trajectory deviation (deg)'

g = sns.relplot(
    kind='scatter',
    data = toplt.sample(n=4000),
    row='cond1',
    col = 'cond0',
    hue = 'traj_dir',
    col_order = all_cond0,
    row_order = all_cond1,
    x = x,
    y = y,
    alpha=0.1,
    linewidth = 0,
    color = 'grey',
    height=3,
    aspect=2/2,
    legend=False
    )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=binned_df.loc[(binned_df['cond0']==all_cond0[j]) & 
                                        (binned_df['cond1']==all_cond1[i])], 
                    x=x, y=y, 
                    hue='cond1',alpha=1,
                    legend=False,
                    ax=ax)

g.set(ylim=(-12,16))
g.set(xlim=(lower,upper))
g.set(xlabel=x+" (deg)")
g.set(ylabel=y+" (deg)")

# g.set_axis_labels(x_var = xlabel, y_var = ylabel)
sns.despine()
plt.savefig(fig_dir+f"/{x} {y} correlation.pdf",format='PDF')
# r_val = stats.pearsonr(toplt[x],toplt[y])[0]
# print(f"pearson's r = {r_val}")

# %% fit sigmoid - master
all_coef = pd.DataFrame()
all_y = pd.DataFrame()
all_binned_average = pd.DataFrame()


for (cond1,cond0,cond_ztime), for_fit in df_toplt.groupby(['cond1','cond0','ztime']):
    if if_jackknife:
        expNum = for_fit['expNum'].max()
        grouped_idx_for_loop = jackknife_resampling(np.array(list(range(expNum+1))))
    else:
        unique_rep = for_fit['expNum'].unique()
        grouped_idx_for_loop = [[rep] for rep in unique_rep]
        
    for repNum, idx_group in enumerate(grouped_idx_for_loop):
        coef, fitted_y, sigma = sigmoid_fit(
            for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE, func=sigfunc_4free
        )
        if ~if_jackknife:
            repNum = idx_group[0]
        slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
        fitted_y.columns = ['Attack angle','rot to angvel max']
        all_y = pd.concat([all_y, fitted_y.assign(
            cond0=cond0,
            cond1=cond1,
            repNum = repNum,
            ztime=cond_ztime,
            )])
        all_coef = pd.concat([all_coef, coef.assign(
            slope=slope,
            cond0=cond0,
            cond1=cond1,
            repNum = repNum,
            ztime=cond_ztime,
            )])
    binned_df = distribution_binned_average(for_fit,by_col=which_rotation,bin_col=which_atk_ang,bin=AVERAGE_BIN)
    binned_df.columns=['rot to angvel max',which_atk_ang]
    all_binned_average = pd.concat([all_binned_average,binned_df.assign(
        cond0=cond0,
        cond1=cond1,
        ztime=cond_ztime,
        )],ignore_index=True)
    
all_y = all_y.reset_index(drop=True)
all_coef = all_coef.reset_index(drop=True)
all_coef.columns=['k','xval','min','height',
                  'slope','cond0','cond1','repNum','ztime']
all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()

# %%

####################################
###### Plotting Starts Here ######
####################################

# plot bout frequency vs IBI pitch and fit with parabola
defaultPlotting(size=12)

plt.figure()

g = sns.relplot(x='rot to angvel max',y='Attack angle', data=all_y, 
                kind='line',
                col='cond0', col_order=all_cond0,
                row = 'ztime', row_order=all_ztime,
                hue='cond1', hue_order = all_cond1, errorbar='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=all_binned_average.loc[
            (all_binned_average['cond0']==all_cond0[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                ], 
                    x='rot to angvel max', y=which_atk_ang, 
                    hue='cond1',alpha=0.5,
                    ax=ax)
upper = np.percentile(df_toplt[which_atk_ang], 95)
lower = np.percentile(df_toplt[which_atk_ang], 5)
g.set(ylim=(lower, upper))

filename = os.path.join(fig_dir,"attack angle vs rot to angvel max.pdf")
plt.savefig(filename,format='PDF')

# plt.show()

# %%
# plot coefs
defaultPlotting(size=12)

# %%
toplt = all_coef
columns_toplt = ['slope','k','xval','min','height','slope']

x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'
units = 'repNum'

for feature in columns_toplt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        aspect = 0.6,
        )
    filename = os.path.join(fig_dir,f"{feature}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
    