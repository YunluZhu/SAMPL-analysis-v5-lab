'''
Attack angles
https://elifesciences.org/articles/45839

This script takes attack angles and pre-bout posture change and calculats the fin-body ratio using sigmoid fit
The sigmoid fit reuqires the max number of bouts. 
Please this version of the code is HARD-CODED, but the sigmoid fit requqires parameter adjustments according to your specific data anyway. 
Will make it softcoded in future versions.

This analysis only takes one variable condition, either experimental groups (sibs, tau, lesion, etc.) or age (dpf, wpf...)

NOTE if frame rate is different than 40Hz, change frame idx used for calculation of pre bout change/pre bout ang/righting rotation... accordingly
NOTE 100ms pre and post is used for ALL calculation. Change if needed

'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
import numpy as np # numpy
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

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# from statannot import add_stat_annotation  # pip install only. if using conda env, run <conda install pip> first

# %%
# Paste root directory here
pick_data = 'hets'
root = get_data_dir(pick_data)

folder_name = f'{pick_data}_tmp_boutProperties3jackPairedComparison'
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.mkdir(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
# CONSTANTS

CLIMB_MIN = 20

BINNED_NUM = 100

X_RANGE = np.arange(-10,15.01,0.01)
BIN_WIDTH = 0.8  
AVERAGE_BIN = np.arange(-10,15,BIN_WIDTH)

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
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
hue_order = list()
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
                raw_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                angles = raw_data.loc[:,['propBoutAligned_pitch','propBoutAligned_speed','propBoutAligned_angVel']]
                angles = angles.assign(idx=int(len(angles)/51)*list(range(0,51)))
                peak_angles = angles.loc[angles['idx']==30]
                peak_angles = peak_angles.assign(
                    time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
                    traj = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['epochBouts_trajectory'].values,
                    )  # peak angle time and bout traj
                peak_angles_day = day_night_split(peak_angles, 'time')
                # peak_angles_day = peak_angles_day.dropna()
                
                # get indices of bout peak (for posture change calculation)
                peak_idx = peak_angles_day.index
                # calculate individual attack angles (traj - pitch)
                # atk_ang = peak_angles_day['propBoutAligned_traj'] - peak_angles_day['propBoutAligned_pitch']
                atk_ang = peak_angles_day['traj'] - angles.loc[peak_idx-2,'propBoutAligned_pitch'].values # new atk angles calculated using accel ang

                # calculate posture change calculation. NOTE change if frame rate changes
                pre_posture_chg = angles.loc[peak_idx-2, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                post_posture_chg = angles.loc[peak_idx+4, 'propBoutAligned_pitch'].values - angles.loc[peak_idx+2, 'propBoutAligned_pitch']

                # try 100ms after peak, NOTE change if frame rate changes
                righting_rot = angles.loc[peak_idx+4, 'propBoutAligned_pitch'].values - angles.loc[peak_idx, 'propBoutAligned_pitch']
                steering_rot = angles.loc[peak_idx, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                
                output_forBout = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                             'pre_posture_chg':pre_posture_chg.values, 
                                             'post_posture_chg':post_posture_chg.values,
                                             'pre_pitch': angles.loc[peak_idx-4,'propBoutAligned_pitch'].values, # try 100ms before peak
                                             'end_pitch': angles.loc[peak_idx+4,'propBoutAligned_pitch'].values, # try 100ms after peak
                                             'accel_rot' : steering_rot.values,
                                             'decel_rot': righting_rot.values,
                                             'traj': peak_angles_day['traj'], 
                                             'pitch': peak_angles_day['propBoutAligned_pitch'],
                                             'speed': angles.loc[peak_idx, 'propBoutAligned_speed'].values,
                                             'accel_ang': angles.loc[peak_idx-2,'propBoutAligned_pitch'].values,
                                             'decel_ang': angles.loc[peak_idx+2,'propBoutAligned_pitch'].values,  # mid bout angel decel
                                             'angVel': angles.loc[peak_idx-1,'propBoutAligned_angVel'].values,
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
            
            mean_data_cond = pd.concat([mean_data_cond, mean_data.assign(dpf=all_conditions[condition_idx][0],
                                                                         condition=all_conditions[condition_idx][4:])])
            all_data_cond = pd.concat([all_data_cond, all_bouts_data.assign(dpf=all_conditions[condition_idx][0],
                                                                         condition=all_conditions[condition_idx][4:])])
            hue_order.append(all_conditions[condition_idx][4:])
# %%

steep_data = all_data_cond.loc[all_data_cond['traj']>20,:]
hue_order.sort()
all_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)            
mean_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)   

# %% ------------OTHER data -----------------

dive_data = all_data_cond.loc[((all_data_cond['traj']<0) & (all_data_cond['pitch']<0)),:]
rise_data = all_data_cond.loc[((all_data_cond['traj']>0) & (all_data_cond['pitch']>0)),:]

negPitch_data = all_data_cond.loc[all_data_cond['pitch']<0,:]
posPitch_data = all_data_cond.loc[all_data_cond['pitch']>0,:]

righting_data = all_data_cond.loc[abs(all_data_cond['pre_pitch'])>abs(all_data_cond['end_pitch']),:] 

# diveCross_data = all_data_cond.loc[((all_data_cond['heading']<0) & (all_data_cond['pitch']>0)),:]
# riseCross_data = all_data_cond.loc[((all_data_cond['heading']>0) & (all_data_cond['pitch']0)),:]

# negPitch_adj = all_data_cond.loc[all_data_cond['pitch']<0 & ,:]

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

y_feature = 'pre_posture_chg'
x_feature = 'accel_ang'


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
elif pick_df == 'righting_data':
    df_sel = righting_data
    
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
fig2.savefig(fig_dir+f"/{pick_data}'s {pick_df} 2 feature {y_feature}_{x_feature}.pdf",format='PDF')

#-----------------------

# plt_dpf = ['7','7']

# for i in range(2):
#     df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
#     print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
#     sns.jointplot(x=x_feature, y=y_feature,data=df_to_plot, kind="kde", height=5, space=0)
#     # plt.show()
 # %%
