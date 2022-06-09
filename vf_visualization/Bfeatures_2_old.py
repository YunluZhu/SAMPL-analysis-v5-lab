'''
Attack angles
https://elifesciences.org/articles/45839

This script takes attack angles and pre-bout posture change and calculats the fin-body ratio using sigmoid fit
The sigmoid fit reuqires the max number of bouts. 
Please this version of the code is HARD-CODED, but the sigmoid fit requqires parameter adjustments according to your specific data anyway. 
Will make it softcoded in future versions.

This analysis only takes one variable condition, either experimental groups (sibs, tau, lesion, etc.) or age (dpf, wpf...)

NOTE 100ms pre and post is used for ALL calculation. Change if needed

'''

#%%
import sys
import os,glob
import time
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
import scipy

from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)


from plot_functions.bout_properties_2_plt import (plt_KDE, plt_meanData)
from plot_functions.get_data_dir import get_data_dir
from plot_functions.get_index import get_index

# from vf_visualization.bout_properties_1_timed_spd_diveVclimb import FRAME_RATE

# %%
# Paste root directory here
pick_data = 'tau_long'
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)

folder_name = f'{pick_data}_boutProperties2_features'
folder_dir = f'/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/{pick_data}'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')

plotKDE = 0
plotMeanData = 0
# %%
# CONSTANTS

CLIMB_MIN = 20

BINNED_NUM = 100
X_RANGE = np.arange(-10,15.01,0.01)
BIN_WIDTH = 0.8  
AVERAGE_BIN = np.arange(-10,15,BIN_WIDTH)
time50ms = int(0.05 * FRAME_RATE)
time100ms = int(0.1 * FRAME_RATE)

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    df_night = df.loc[hour[(hour<9) | (hour>23)].index, :]
    return df_day

def distribution_binned_average(df, bin_width, condition):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0:2],condition=condition[4:])
    return df_out

# %%
# get data 
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


# jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
# jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

all_data_cond = pd.DataFrame()
mean_data_cond = pd.DataFrame()
all_cond1 = []
all_cond2 = []
# binned_atk_angles = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_bouts_data = pd.DataFrame()
            mean_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # - attack angle calculation
                exp_path = os.path.join(subpath, exp)
                angles = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned').loc[:,['propBoutAligned_pitch','propBoutAligned_speed']]
                angles = angles.assign(idx=int(len(angles)/total_aligned)*list(range(0,total_aligned)))
                peak_angles = angles.loc[angles['idx']==peak_idx]
                peak_angles = peak_angles.assign(
                    time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
                    traj = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['epochBouts_trajectory'].values,
                    )  # peak angle time and bout traj
                peak_angles_day = day_night_split(peak_angles, 'time')
                # peak_angles_day = peak_angles_day.dropna()
                
                # get indices of bout peak (for posture change calculation)
                all_peak_idx = peak_angles_day.index
                # calculate individual attack angles (traj - pitch)
                # atk_ang = peak_angles_day['propBoutAligned_traj'] - peak_angles_day['propBoutAligned_pitch']
                atk_ang = peak_angles_day['traj'] - angles.loc[all_peak_idx-time50ms,'propBoutAligned_pitch'].values # new atk angles calculated using accel ang

                # calculate posture change calculation. NOTE change if frame rate changes
                pre_posture_chg = angles.loc[all_peak_idx-time50ms, 'propBoutAligned_pitch'].values - angles.loc[all_peak_idx-time100ms, 'propBoutAligned_pitch']
                post_posture_chg = angles.loc[all_peak_idx+time100ms, 'propBoutAligned_pitch'].values - angles.loc[all_peak_idx+time50ms, 'propBoutAligned_pitch']

                # try 100ms after peak, NOTE change if frame rate changes
                righting_rot = angles.loc[all_peak_idx+time100ms, 'propBoutAligned_pitch'].values - angles.loc[all_peak_idx, 'propBoutAligned_pitch']
                steering_rot = angles.loc[all_peak_idx, 'propBoutAligned_pitch'].values - angles.loc[all_peak_idx-time100ms, 'propBoutAligned_pitch']
                
                output_forBout = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                             'pre_posture_chg':pre_posture_chg.values, 
                                             'post_posture_chg':post_posture_chg.values,
                                             'pre_pitch': angles.loc[all_peak_idx-time100ms,'propBoutAligned_pitch'].values, # try 100ms before peak
                                             'end_pitch': angles.loc[all_peak_idx+time100ms,'propBoutAligned_pitch'].values, # try 100ms after peak
                                             'accel_rot' : steering_rot.values,
                                             'decel_rot': righting_rot.values,
                                             'traj': peak_angles_day['traj'], 
                                             'pitch': peak_angles_day['propBoutAligned_pitch'],
                                             'speed': angles.loc[all_peak_idx, 'propBoutAligned_speed'].values,
                                             'accel_ang': angles.loc[all_peak_idx-time50ms,'propBoutAligned_pitch'].values,
                                             'decel_ang': angles.loc[all_peak_idx+time50ms,'propBoutAligned_pitch'].values,  # mid bout angel decel
                                             'expNum':[expNum]*len(pre_posture_chg),
                                             'date':exp[0:6]})
                output_forBout = output_forBout.assign(angle_chg = output_forBout.end_pitch - output_forBout.pre_pitch)
                all_bouts_data = pd.concat([all_bouts_data, output_forBout], ignore_index=True)             
                # calculate and concat mean posture change (mean rotation) and mean of atk_ang of the current experiment
                # mean of each experiment
                mean_data = pd.concat([mean_data, pd.DataFrame(data={'atkAng':np.nanmean(atk_ang),
                                                                     'maxSpd':np.nanmean(peak_angles_day['propBoutAligned_speed']),
                                                                     'meanRot':np.nanmean(pre_posture_chg),
                                                                     'date':exp[0:6]
                                                                     }, index=[expNum])])
                # end of exp loop
            cond1 = all_conditions[condition_idx].split("_")[0]
            cond2 = all_conditions[condition_idx].split("_")[1]
            all_cond1.append(cond1)
            all_cond2.append(cond2)

            mean_data_cond = pd.concat([mean_data_cond, mean_data.assign(dpf=cond1,
                                                                        condition=cond2)])
            all_data_cond = pd.concat([all_data_cond, all_bouts_data.assign(dpf=cond1,
                                                                        condition=cond2)])
all_cond1 = list(set(all_cond1))
all_cond1.sort()
all_cond2 = list(set(all_cond2))
all_cond2.sort()
hue_order = all_cond2
# %%
steep_data = all_data_cond.loc[all_data_cond['traj']>20,:]
all_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)            
mean_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)   

# %% mean data
if plotMeanData == 1:
    plt_meanData(mean_data_cond,hue_order)

# %% KDE plots
if plotKDE == 1:
    plt_KDE(all_data_cond,hue_order)
    
# %%
# Pre bout and decel, accel
current_palette = sns.color_palette()

# pre_pitch - decel rot 
flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())
df = all_data_cond
plt_condition = hue_order
fig_num = len(plt_condition)

fig, axes = plt.subplots(fig_num,sharey=True,sharex=True)
fig.set_figheight(4*fig_num)

fig2, axes2 = plt.subplots(2)
fig2.set_figheight(8)

for i, cur_cond in enumerate(plt_condition):
    df_to_plot = df.loc[df['condition']==cur_cond,:]
    print(f'{plt_condition[i]}')
    sns.kdeplot(df_to_plot['pre_pitch'], df_to_plot['decel_rot'], ax=axes[i], shade=True)
    
    df_to_plot = df_to_plot.sort_values(by = 'pre_pitch')
    df_binned = df_to_plot.groupby(np.arange(len(df_to_plot))//BINNED_NUM).mean()
    sns.lineplot(x = 'pre_pitch', y = 'decel_rot',data=df_binned,ax=axes[i],alpha=1,color="red")

    sns.lineplot(x = 'pre_pitch', y = 'decel_rot',data=df_binned,ax=axes2[0],alpha=0.7,
                 palette=current_palette[i])

# accel - righting rot
flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())
df = all_data_cond
plt_condition = hue_order
fig_num = len(plt_condition)

fig3, axes3 = plt.subplots(fig_num,sharey=True,sharex=True)
fig3.set_figheight(4*fig_num)

for i, cur_cond in enumerate(plt_condition):
    df_to_plot = df.loc[df['condition']==cur_cond,:]
    print(f'{plt_condition[i]}')
    sns.kdeplot(df_to_plot['accel_rot'], df_to_plot['decel_rot'], ax=axes3[i], shade=True)
    
    df_to_plot = df_to_plot.sort_values(by = 'accel_rot')
    df_binned = df_to_plot.groupby(np.arange(len(df_to_plot))//BINNED_NUM).mean()
    sns.lineplot(x = 'accel_rot', y = 'decel_rot',data=df_binned,ax=axes3[i],alpha=1,color="red")

    sns.lineplot(x = 'accel_rot', y = 'decel_rot',data=df_binned,ax=axes2[1],alpha=0.7,
                 palette=current_palette[i])
    
fig2.legend(labels=plt_condition)

plt.tight_layout()

plt.show()

fig.savefig(fig_dir+"/righting_prePitch.pdf",format='PDF')
fig2.savefig(fig_dir+"/righting_steering_prePitch_across_cond.pdf",format='PDF')
fig3.savefig(fig_dir+"/righting_steering.pdf",format='PDF')

# %% linear regression

# fig4, axes4 = plt.subplots(2,sharey=True,sharex=True)
# fig4.set_figheight(4*2)

# for i, cur_cond in enumerate(plt_condition):
#     df_to_plot = df.loc[df['condition']==cur_cond,:]
#     print(f'{plt_condition[i]}')
#     x = df_to_plot['accel_rot']
#     y = df_to_plot['decel_rot']
#     coef = np.polyfit(x,y,1)
#     print(coef)
#     poly1d_fn = np.poly1d(coef) 

#     # axes4[i].plot(x,y, 'yo', x, poly1d_fn(x), '--k')
#     sns.scatterplot(x=x,y=y, ax=axes4[i], alpha=0.01)
#     sns.lineplot(x, poly1d_fn(x),ax=axes4[i])
#     # df_to_plot = df_to_plot.sort_values(by = 'accel_rot')
#     # df_binned = df_to_plot.groupby(np.arange(len(df_to_plot))//BINNED_NUM).mean()
#     # sns.lineplot(x = 'accel_rot', y = 'decel_rot',data=df_binned,ax=axes3[i],alpha=1,color="red")

#     # sns.lineplot(x = 'accel_rot', y = 'decel_rot',data=df_binned,ax=axes2[1],alpha=0.7,
#     #              palette=current_palette[i])


# %%
# Cumulative fractions of postures during climbs with trajectories greater than 20° (Figure 1E)

df = steep_data
defaultPlotting()
current_palette = sns.color_palette()

data_7S = df.loc[(df['dpf']=='7') & (df['condition']=='Sibs'),:]
data_7T = df.loc[(df['dpf']=='7') & (df['condition']=='Tau'),:]

p = sns.kdeplot(data=data_7S['pitch'],cumulative=True,color=current_palette[0],linewidth=2,label="day7_Sibs")
p = sns.kdeplot(data=data_7T['pitch'],cumulative=True,color=current_palette[1],linewidth=2,label="day7_Tau")

data_4S = df.loc[(df['dpf']=='4') & (df['condition']=='Sibs'),:]
data_4T = df.loc[(df['dpf']=='4') & (df['condition']=='Tau'),:]

p = sns.kdeplot(data=data_4S['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[0],label="day4_Sibs")
p = sns.kdeplot(data=data_4T['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[1],label="day4_Sibs")

plt.ylabel("Cumulative fractions")
plt.xlabel("Postures during climbs (deg)")

p.set_xlim(-10,90)




# %% ------------OTHER data -----------------

# dive_data = all_data_cond.loc[all_data_cond['angle_chg']<0,:]
# rise_data = all_data_cond.loc[all_data_cond['angle_chg']>0,:]

dive_data = all_data_cond.loc[((all_data_cond['traj']<0) & (all_data_cond['pitch']<0)),:]
rise_data = all_data_cond.loc[((all_data_cond['traj']>0) & (all_data_cond['pitch']>0)),:]

negPitch_data = all_data_cond.loc[all_data_cond['pitch']<0,:]
posPitch_data = all_data_cond.loc[all_data_cond['pitch']>0,:]

tmp_data = all_data_cond.loc[all_data_cond['traj']<-10,:]

# diveCross_data = all_data_cond.loc[((all_data_cond['traj']<0) & (all_data_cond['pitch']>0)),:]
# riseCross_data = all_data_cond.loc[((all_data_cond['traj']>0) & (all_data_cond['pitch']0)),:]

# negPitch_adj = all_data_cond.loc[all_data_cond['pitch']<0 & ,:]

# %%
# Which to plot
# or all_data_cond or steep_data
pick_df = 'all'


if pick_df == 'dive_data':
    df_sel = dive_data
elif pick_df == 'rise_data':
    df_sel = rise_data
elif pick_df == 'negPitch_data':
    df_sel = negPitch_data
elif pick_df =='posPitch_data':
    df_sel = posPitch_data
elif pick_df == 'all':
    df_sel = all_data_cond

# mean data ---------------------------------

plt_condition = hue_order
df_mean = pd.DataFrame()
cat_cols = ['condition','expNum','dpf']
df_mean = df_sel.groupby(cat_cols).median().reset_index()


defaultPlotting()
fig_col_number = 3
fig_num = df_mean.shape[1]-3
fig, axes = plt.subplots(round(fig_num/fig_col_number),fig_col_number)
fig.set_figheight(len(axes)/fig_col_number*fig_num)
fig.set_figwidth(8)

ttest_p = list()
ci_dic = {1:None,2:"sd",3:"sd"}
for i, ax in enumerate(axes.flatten()):
    if i+1<=fig_num:
        if len(all_cond1) == 1: # if only one cond1, plot indivial repeats
            g = sns.pointplot(x='condition',y=df_mean.iloc[:,i+len(cat_cols)], hue='date',data=df_mean,
                        palette=sns.color_palette(flatui), scale=0.5,
                        #   order=['Sibs','Tau','Lesion'],
                        ax=ax)
        g = sns.pointplot(x='condition', y=df_mean.iloc[:,i+len(cat_cols)], data=df_mean, 
                          order = all_cond2,
                          hue='dpf',
                          hue_order = all_cond1,
                          linewidth=0,
                          alpha=0.9,
                          dodge=bool(len(all_cond1)),
                          ci=ci_dic[len(all_cond1)],
                          markers=['d','x'],
                          ax=ax)
        # p-value calculation

        ttest_res, pval = ttest_rel(df_mean.loc[df_mean['condition']==plt_condition[0],df_mean.columns[i+len(cat_cols)]],
                                    df_mean.loc[df_mean['condition']==plt_condition[1],df_mean.columns[i+len(cat_cols)]])
        ttest_p.append(pval)
        # g.legend_.remove()

plt.tight_layout()

plt.savefig(fig_dir+f"/{pick_data}_{pick_df}_bouts properties.pdf",format='PDF')
plt.show()
print(f'{pick_data} - Sibs v.s. Tau: paired t-test p-value = \n{ttest_p}')

# %% 2 feature plot
# 'atk_ang',
# 'pre_posture_chg',
# 'post_posture_chg',
# 'traj',
# 'pitch',
# 'pre_pitch',
# 'end_pitch',
# 'decel_rot',
# 'accel_rot',
# 'decel_ang',
# 'accel_ang',
# 'speed'
# 'angle_chg'

pick_df = 'all'

y_feature = 'traj'
x_feature = 'pre_pitch'


if pick_df == 'dive_data':
    df_sel = dive_data
elif pick_df == 'rise_data':
    df_sel = rise_data
elif pick_df == 'negPitch_data':
    df_sel = negPitch_data
elif pick_df =='posPitch_data':
    df_sel = posPitch_data
elif pick_df == 'all':
    df_sel = all_data_cond
    
df2 = df_sel

current_palette = sns.color_palette()

# pre_pitch - decel rot 
flatui = ["#D0D0D0"] * (df2.groupby('condition').size().max())
df = df2
plt_condition = hue_order
fig_num = len(plt_condition)

fig, axes = plt.subplots(fig_num,sharey=True,sharex=True)
fig.set_figheight(4*fig_num)

fig2, axes2 = plt.subplots(1)
fig2.set_figheight(4)

for i, cur_cond in enumerate(plt_condition):
    df_to_plot = df.loc[df['condition']==cur_cond,:]
    print(f'{plt_condition[i]}')
    sns.kdeplot(x = x_feature, y = y_feature,data=df_to_plot, ax=axes[i], shade=True)
    
    df_to_plot = df_to_plot.sort_values(by = x_feature)
    df_binned = df_to_plot.groupby(np.arange(len(df_to_plot))//BINNED_NUM).mean()
    sns.lineplot(x = x_feature, y = y_feature,data=df_binned,ax=axes[i],alpha=1,color="red")

    sns.lineplot(x = x_feature, y = y_feature,data=df_binned,ax=axes2,alpha=0.7,
                 palette=current_palette[i])
    
    
fig2.legend(labels=plt_condition)

plt.tight_layout()

plt.show()
fig2.savefig(fig_dir+f"/{pick_data}_{pick_df} 2 feature {y_feature}_{x_feature}.pdf",format='PDF')

#-----------------------

# plt_dpf = ['7','7']

# for i in range(2):
#     df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
#     print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
#     sns.jointplot(x=x_feature, y=y_feature,data=df_to_plot, kind="kde", height=5, space=0)
#     # plt.show()
 # %%
