'''
Fin-body ratio with new definitions slightly different from eLife 2019. Works well.

plot attack angle vs. pre bout rotation, fit with a sigmoid w/ 4-free parameters

zeitgeber time? Yes
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
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average, linReg_sampleSatter_plot)
from scipy import stats

set_font_type()
defaultPlotting(size=16)
# %%
pick_data = 'wt_fin'
which_zeitgeber = 'day' # day / night / all
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 500
which_rot = 'rot_to_maxAngvel'
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
# Select data and create figure foldeor
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
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# # %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# if FRAME_RATE > 100:
#     all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
# elif FRAME_RATE == 40:
#     all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<4].index, inplace=True)
all_feature_cond = all_feature_cond.assign(
    body_angvel = all_feature_cond[which_rot]/0.2  ,
    fin_angvel = all_feature_cond['rot_late_accel']/0.05,
    direction = pd.cut(all_feature_cond['pitch_pre_bout'],bins=[-90,10,90],labels=['noseDN','noseUP']),
    traj_direction = pd.cut(all_feature_cond['traj_peak'],bins=[-90,0,90],labels=['trajDN','trajUP']),
)

# %%
filtered_data = all_feature_cond.drop(all_feature_cond.loc[(all_feature_cond['atk_ang']<0) & (all_feature_cond['rot_full_accel']>0)].index)
filtered_data = filtered_data.loc[filtered_data['spd_peak']>7]

IBI_angles_day_resampled = pd.DataFrame()
IBI_angles_night_resampled = pd.DataFrame()

if which_zeitgeber != 'night':
    IBI_angles_day_resampled = filtered_data.loc[
        filtered_data['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )
if which_zeitgeber != 'day':
    IBI_angles_night_resampled = filtered_data.loc[
        filtered_data['ztime']=='night',:
            ]
    if NIGHT_RESAMPLE != 0:  # if resampled
        IBI_angles_night_resampled = IBI_angles_night_resampled.groupby(
                ['cond0','cond1','expNum']
                ).sample(
                        n=NIGHT_RESAMPLE,
                        replace=True
                        )
all_feature_cond_resampled = pd.concat([IBI_angles_day_resampled,IBI_angles_night_resampled],ignore_index=True)


# %%
toplt = all_feature_cond_resampled

plt_dict = {
    'early_rotation vs atk_ang':[which_rot,'atk_ang'],
    'late_rotation vs atk_ang':['rot_late_accel','atk_ang'],
    # 'late_rotation vs early_rotation':['rot_late_accel',which_rot],
    # 'early_rotation vs traj':['atk_ang','traj_peak'],

    # 'atk_ang vs early_rotation':['atk_ang',which_rot],

}
what_cond1 = 'cond 7'
for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 1
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
        data = toplt.sample(frac=0.5),
        row='cond1',
        col = 'cond0',
        col_order = all_cond0,
        row_order = all_cond0,

        # facet_kws={'sharey': False, 'sharex': False},
        
        x = x,
        y = y,
        # x_bins=np.arange(round_half_up(lower),round_half_up(upper),3),
        # x_ci=95,
        alpha=0.1,
        # hue='direction',
        # marker='+',
        linewidth = 0,
        color = 'grey',
        height=3,
        aspect=2/2,
        )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=binned_df.loc[(binned_df['cond0']==all_cond0[j]) & 
                                            (binned_df['cond1']==all_cond0[i])], 
                        x=x, y=y, 
                        hue='cond1',alpha=1,
                        ax=ax)
    
    g.set(ylim=(-15,20))
    g.set(xlim=(lower,upper))
    
    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} {what_cond1}.pdf",format='PDF')
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

# corr of two rotations, separate by direction
which_dir = 'direction'
toplt = all_feature_cond_resampled.query('spd_peak >= 7')
plt_dict = {
    'early_rotation vs atk_ang':[which_rot,'atk_ang'],
    'early_rotation vs late_rotation':[which_rot,'rot_late_accel'],
        'traj_deviation vs atk_ang':['traj_deviation','atk_ang'],

}
what_cond1 = 'dir 7'
for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 0.5
    AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
    binned_df = toplt.groupby([which_dir,'cond1','cond0']).apply(
        lambda group: distribution_binned_average(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
    )
    binned_df.columns=[x,y]
    binned_df = binned_df.reset_index(level=[which_dir,'cond1','cond0'])
    binned_df = binned_df.reset_index(drop=True)
    # xlabel = "Relative pitch change (deg)"
    # ylabel = 'Trajectory deviation (deg)'
 
    g = sns.relplot(
        kind='scatter',
        data = toplt.sample(frac=0.2),
        x = x,
        y = y,
        hue=which_dir,
        row='cond1',
        row_order = all_cond0,
        col = 'cond0',
        col_order = all_cond0,
        # x_bins=np.arange(round_half_up(lower),round_half_up(upper),3),
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

    g.set(ylim=(np.percentile(toplt[y], 1),np.percentile(toplt[y], 99.5)))
    g.set(xlim=(lower,upper))
    
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=binned_df.loc[(binned_df['cond0']==all_cond0[j]) & 
                                            (binned_df['cond1']==all_cond0[i])], 
                         hue=which_dir,
                        x=x, y=y, 
                        alpha=1,
                        ax=ax)
    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} {what_cond1}.pdf",format='PDF')
    r_val = stats.pearsonr(toplt[x],toplt[y])[0]
    print(f"pearson's r = {r_val}")
# %%


# others to plot for testing
# %%
toplt = all_feature_cond_resampled.query('spd_peak >= 7' 
                                         )
plt_dict = {
    # 'atk_ang vs early_rotation':['atk_ang',which_rot],

    # 'early_rotation vs traj_Deviation':[which_rot,'traj_deviation'],
    # 'rot_full_accel_phased vs traj_Deviation':['rot_full_accel_phased','traj_deviation'],
    'traj_deviation vs atk_ang':['traj_deviation','atk_ang'],

}
what_cond1 = 'cond 7'
for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 1
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
        data = toplt.sample(frac=0.5),
        row='cond1',
        col = 'cond0',
        col_order = all_cond0,
        row_order = all_cond0,

        x = x,
        y = y,
        # x_bins=np.arange(round_half_up(lower),round_half_up(upper),3),
        # x_ci=95,
        alpha=0.1,
        # hue='direction',
        # marker='+',
        linewidth = 0,
        color = 'grey',
        height=3,
        aspect=2/2,
        )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=binned_df.loc[(binned_df['cond0']==all_cond0[j]) & 
                                            (binned_df['cond1']==all_cond0[i])], 
                        x=x, y=y, 
                        hue='cond1',alpha=1,
                        ax=ax)
    
    g.set(ylim=(-15,20))
    g.set(xlim=(lower,upper))
    

    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} {what_cond1}.pdf",format='PDF')
    r_val = stats.pearsonr(toplt[x],toplt[y])[0]
    print(f"pearson's r = {r_val}")
# %%
# linear fit of atk_ang vs early rotation up dn separated
def func(x,k,b):
    y = k * x + b
    return y

all_coef = pd.DataFrame()
all_y = pd.DataFrame()

df_tofit = all_feature_cond

for (cond1,cond0,this_ztime), for_fit in df_tofit.groupby(['cond1','cond0','ztime']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_for_fit = for_fit.loc[for_fit['expNum'].isin(idx_group)]
        k, y_intersect = np.polyfit(x = this_for_fit['pitch_peak'], 
                            y = this_for_fit['traj_peak'],
                            deg = 1) 
        x_intersect = -1 * y_intersect / k
        all_coef = pd.concat([all_coef, pd.DataFrame(
            data = {
                'k': k,
                'y_intersect': y_intersect,
                'x_intersect': x_intersect,
                'cond0': cond0,
                'cond1': cond1,
                'excluded_exp': excluded_exp,
                'ztime': this_ztime,
            }, index=[0]
        )
                            ], ignore_index=True)
        
        y = func(X_RANGE,k, y_intersect)
        this_y = pd.DataFrame(data=y, columns=['y']).assign(
            x=X_RANGE,
            cond0 = cond0,
            cond1 = cond1,
            excluded_exp = excluded_exp,
            ztime = this_ztime,)
        all_y = pd.concat([all_y, this_y], ignore_index=True)

all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()

# plot fitted line
plt.figure()

g = sns.relplot(x='x',y='y', data=all_y, 
                kind='line',
                col='cond0',
                # row = 'cond1', 
                hue='cond1',
                ci='sd',
                )
# g.set(xlim=(-30, 40))
# g.set(ylim=(-30, 40))

filename = os.path.join(fig_dir,"fin-body line reg.pdf")
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
    filename = os.path.join(fig_dir,f"fin-body line reg coef by age {feature}.pdf")
    plt.savefig(filename,format='PDF')
# %%
