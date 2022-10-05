'''
Fin-body ratio with new definitions slightly different from eLife 2019. Works well.

plot attack angle vs. pre bout rotation, fit with a sigmoid w/ 4-free parameters

zeitgeber time? Yes
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
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average, linReg_sampleSatter_plot)
from scipy import stats

set_font_type()
defaultPlotting(size=16)
# %%
pick_data = '7dd_all'
which_zeitgeber = 'day' # day / night / all
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 500
which_rot = 'rot_mid_accel_initial'
# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,    0,    -20,   1]
    upper_bounds = [10,      5,     2,      50]
    x0=[1, 1, -1, 10]
    
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
    popt, pcov = curve_fit(func, df[which_rot], df['atk_ang'], 
                        #    maxfev=2000, 
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

X_RANGE = np.arange(-5,5.01,0.01)
BIN_WIDTH = 0.3
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK2_fin_body_correlation_z{which_zeitgeber}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# # %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# if FRAME_RATE > 100:
#     all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
# elif FRAME_RATE == 40:
#     all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<4].index, inplace=True)


# %%

which_rot = 'rot_mid_accel_initial'


IBI_angles_day_resampled = pd.DataFrame()
IBI_angles_night_resampled = pd.DataFrame()

if which_zeitgeber != 'night':
    IBI_angles_day_resampled = all_feature_cond.loc[
        all_feature_cond['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                ['dpf','condition','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )
if which_zeitgeber != 'day':
    IBI_angles_night_resampled = all_feature_cond.loc[
        all_feature_cond['ztime']=='night',:
            ]
    if NIGHT_RESAMPLE != 0:  # if resampled
        IBI_angles_night_resampled = IBI_angles_night_resampled.groupby(
                ['dpf','condition','expNum']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True
                        )
all_feature_cond_resampled = pd.concat([IBI_angles_day_resampled,IBI_angles_night_resampled],ignore_index=True)
all_feature_cond_resampled = all_feature_cond_resampled.assign(
    body_angvel = all_feature_cond_resampled['rot_mid_accel_initial']/0.2  ,
    fin_angvel = all_feature_cond_resampled['rot_late_accel']/0.05,
    direction = pd.cut(all_feature_cond_resampled['pitch_pre_bout'],bins=[-90,10,90],labels=['noseDN','noseUP'])
)

# %%
# toplt = all_feature_cond_resampled
toplt = all_feature_cond_resampled.query('spd_peak >= 7' 
                                         )

# %%
plt_dict = {
    'body_rotation vs atk_ang':['rot_mid_accel_initial','atk_ang'],
    'fin_rotation vs atk_ang':['rot_late_accel','atk_ang'],
    # 'body_rotation vs fin_rotation':['rot_mid_accel_initial','rot_late_accel'],

}

for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 0.5
    AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
    binned_df = distribution_binned_average(toplt,by_col=x,bin_col=y,bin=AVERAGE_BIN)
    binned_df.columns=[x,y]

    # xlabel = "Relative pitch change (deg)"
    # ylabel = 'Trajectory deviation (deg)'
 
    g = sns.relplot(
        kind='scatter',
        data = toplt.sample(frac=0.2),
        x = x,
        y = y,
        # x_bins=np.arange(int(lower),int(upper),3),
        # x_ci=95,
        alpha=0.1,
        # hue='direction',
        # marker='+',
        linewidth = 0,
        color = 'grey',
        height=3,
        aspect=2/2,
        )
    # g.set(ylim=(-25,40))

    g.set(ylim=(-15,20))
    g.set(xlim=(lower,upper))
    
    g.map(sns.lineplot,data=binned_df,
        x=x,
        y=y
        )
    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} {input('what condition')}.pdf",format='PDF')
    r_val = stats.pearsonr(toplt[x],toplt[y])[0]
    print(f"pearson's r = {r_val}")


# # %%
# p = sns.lmplot(
#     data = toplt,
#     x = 'body_angvel',
#     y = 'fin_angvel',
#     x_bins= 10
# )
# # p.set(
# #     ylim=(-70,100),
# #     xlim=(-70,100)
# #     )
# %%
# corr of two rotations, separate by speed

toplt = all_feature_cond_resampled.query('spd_peak >= 7' 
                                         )
plt_dict = {
    'body_rotation vs atk_ang':['rot_mid_accel_initial','atk_ang'],
    'fin_rotation vs atk_ang':['rot_late_accel','atk_ang'],
    'body_rotation vs fin_rotation':['rot_mid_accel_initial','rot_late_accel'],
}
what_condition = input('what condition')
for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 0.5
    AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
    binned_df = toplt.groupby('direction').apply(
        lambda group: distribution_binned_average(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
    )
    binned_df.columns=[x,y]
    binned_df = binned_df.reset_index(level='direction')
    binned_df = binned_df.reset_index(drop=True)
    # xlabel = "Relative pitch change (deg)"
    # ylabel = 'Trajectory deviation (deg)'
 
    g = sns.relplot(
        kind='scatter',
        data = toplt.sample(frac=0.2),
        x = x,
        y = y,
        hue='direction',
        # x_bins=np.arange(int(lower),int(upper),3),
        # x_ci=95,
        alpha=0.1,
        # hue='direction',
        # marker='+',
        linewidth = 0,
        color = 'grey',
        height=3,
        aspect=2/2,
        )
    # g.set(ylim=(-25,40))

    g.set(ylim=(np.percentile(toplt[y], 1),np.percentile(toplt[y], 99)))
    g.set(xlim=(lower,upper))
    
    g.map(sns.lineplot,data=binned_df,
        x=x,
        y=y,
        hue='direction'
        )
    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} {what_condition}.pdf",format='PDF')
    r_val = stats.pearsonr(toplt[x],toplt[y])[0]
    print(f"pearson's r = {r_val}")
# %%
