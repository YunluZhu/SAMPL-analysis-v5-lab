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

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# from statannot import add_stat_annotation  # pip install only. if using conda env, run <conda install pip> first

# %%
# Paste root directory here
root = "/Volumes/LabData/tmp_toana"
# root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7DD_hets_resliced/combined_7DD_NTau-hets_data"
fig_dir = "/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/NT_tmp_plots/"
plotKDE = 1
plotMeanData = 0
# %%
# CONSTANTS
HEADING_LIM = 90
# HEADING_LIM = 180

CLIMB_MIN = 20

BINNED_NUM = 100

FIN_BODY_LOW_LIM = -10  # lower pitch change limit to plot fin_body sigmoid fit and binned data
FIN_BODY_UP_LIM = 15  # 

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
                angles = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned').loc[:,['propBoutAligned_heading','propBoutAligned_pitch','propBoutAligned_speed']]
                angles = angles.assign(idx=int(len(angles)/51)*list(range(0,51)))
                peak_angles = angles.loc[angles['idx']==30]
                peak_angles = peak_angles.assign(
                    time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
                    )  # peak angle
                peak_angles_day = day_night_split(peak_angles, 'time')
                # peak_angles_day = peak_angles_day.dropna()
                # filter for angles meet the condition
                peak_angles_day = peak_angles_day.loc[(peak_angles_day['propBoutAligned_heading']<HEADING_LIM) & 
                                                      (peak_angles_day['propBoutAligned_heading']>-HEADING_LIM)]
                
                # calculate individual attack angles (heading - pitch)
                atk_ang = peak_angles_day['propBoutAligned_heading'] - peak_angles_day['propBoutAligned_pitch']
                
                # get indices of bout peak (for posture change calculation)
                peak_idx = peak_angles_day.index
                # calculate posture change calculation. NOTE change if frame rate changes
                pre_posture_chg = angles.loc[peak_idx-2, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                # try 100ms after peak, NOTE change if frame rate changes
                righting_rot = angles.loc[peak_idx+4, 'propBoutAligned_pitch'].values - angles.loc[peak_idx, 'propBoutAligned_pitch']
                steering_rot = angles.loc[peak_idx, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                
                output_forBout = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                             'pre_posture_chg':pre_posture_chg.values, 
                                             'pre_pitch': angles.loc[peak_idx-4,'propBoutAligned_pitch'].values, # try 100ms before peak
                                             'end_pitch': angles.loc[peak_idx+4,'propBoutAligned_pitch'].values, # try 100ms after peak
                                             'accel_rot' : steering_rot.values,
                                             'decel_rot': righting_rot.values,
                                             'heading': peak_angles_day['propBoutAligned_heading'], 
                                             'pitch': peak_angles_day['propBoutAligned_pitch'],
                                             'speed': angles.loc[peak_idx, 'propBoutAligned_speed'].values,
                                             'accel_ang': angles.loc[peak_idx-2,'propBoutAligned_pitch'].values,
                                             'decel_ang': angles.loc[peak_idx+2,'propBoutAligned_pitch'].values,  # mid bout angel decel
                                             'expNum':[expNum]*len(pre_posture_chg),
                                             'date':exp[0:6]})
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

steep_data = all_data_cond.loc[all_data_cond['heading']>20,:]
hue_order.sort()
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

fig.savefig(fig_dir+"righting_prePitch.pdf",format='PDF')
fig2.savefig(fig_dir+"righting_steering_prePitch_across_cond.pdf",format='PDF')
fig3.savefig(fig_dir+"righting_steering.pdf",format='PDF')


# %%
# Cumulative fractions of postures during climbs with trajectories greater than 20° (Figure 1E)

df = steep_data
defaultPlotting()
current_palette = sns.color_palette()

data_7S = df.loc[(df['dpf']=='7') & (df['condition']=='1Sibs'),:]
data_7T = df.loc[(df['dpf']=='7') & (df['condition']=='2Tau'),:]

p = sns.kdeplot(data=data_7S['pitch'],cumulative=True,color=current_palette[0],linewidth=2,label="day7_Sibs")
p = sns.kdeplot(data=data_7T['pitch'],cumulative=True,color=current_palette[1],linewidth=2,label="day7_Tau")

data_4S = df.loc[(df['dpf']=='4') & (df['condition']=='1Sibs'),:]
data_4T = df.loc[(df['dpf']=='4') & (df['condition']=='2Tau'),:]

p = sns.kdeplot(data=data_4S['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[0],label="day4_Sibs")
p = sns.kdeplot(data=data_4T['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[1],label="day4_Sibs")

plt.ylabel("Cumulative fractions")
plt.xlabel("Postures during climbs (deg)")

p.set_xlim(-10,90)




# %% ------------OTHER data -----------------

dive_data = all_data_cond.loc[((all_data_cond['heading']<0) & (all_data_cond['pitch']<0)),:]
rise_data = all_data_cond.loc[((all_data_cond['heading']>0) & (all_data_cond['pitch']>0)),:]

negPitch_data = all_data_cond.loc[all_data_cond['pitch']<0,:]
posPitch_data = all_data_cond.loc[all_data_cond['pitch']>0,:]

righting_data = all_data_cond.loc[abs(all_data_cond['pre_pitch'])>abs(all_data_cond['end_pitch']),:] 

# diveCross_data = all_data_cond.loc[((all_data_cond['heading']<0) & (all_data_cond['pitch']>0)),:]
# riseCross_data = all_data_cond.loc[((all_data_cond['heading']>0) & (all_data_cond['pitch']0)),:]

# negPitch_adj = all_data_cond.loc[all_data_cond['pitch']<0 & ,:]

# %%
# Which to plot
# or all_data_cond or steep_data
df2 = negPitch_data


# mean data ---------------------------------

plt_condition = hue_order
df_mean = pd.DataFrame()
for condition in plt_condition:
    tmp = df2.loc[df2['condition']==condition ,:]
    mean_data = tmp.groupby('expNum')[['atk_ang',
                                       'pre_posture_chg',
                                       'heading',
                                       'pitch',
                                       'pre_pitch',
                                       'end_pitch',
                                       'decel_rot',
                                       'accel_rot',
                                       'decel_ang',
                                       'accel_ang',
                                       'speed']].apply(
        lambda x: x.median()
    )
    abs_mean_data = tmp.groupby('expNum')['pitch'].apply(
    lambda x: x.abs().mean()
    )
    mean_data = mean_data.assign(deviation_of_posture_from_hori = abs_mean_data.values)
    mean_data = mean_data.assign(condition=condition,
                                 dpf=tmp.groupby('expNum')['dpf'].head(1).values,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_mean = pd.concat([df_mean,mean_data],ignore_index=True)

# multi_comp = MultiComparison(df_mean['atk_ang'], df_mean['dpf']+df_mean['condition'])
# print('* attack angles')
# print(multi_comp.tukeyhsd().summary())
# multi_comp = MultiComparison(df_mean['pre_posture_chg'], df_mean['dpf']+df_mean['condition'])
# print('* max Speed')
# print(multi_comp.tukeyhsd().summary())
# multi_comp = MultiComparison(df_mean['heading'], df_mean['dpf']+df_mean['condition'])
# print('* mean rotation')
# print(multi_comp.tukeyhsd().summary())
# multi_comp = MultiComparison(df_mean['pitch'], df_mean['dpf']+df_mean['condition'])
# print('* mean rotation')
# print(multi_comp.tukeyhsd().summary())

defaultPlotting()
fig_col_number = 3
fig_num = df_mean.shape[1]-3
fig, axes = plt.subplots(round(fig_num/fig_col_number),fig_col_number)
fig.set_figheight(len(axes)/fig_col_number*fig_num)
fig.set_figwidth(8)

ttest_p = list()

for i, ax in enumerate(axes.flatten()):
    if i+1<=fig_num:
        g = sns.pointplot(x='condition',y=df_mean.iloc[:,i], hue='date',data=df_mean,
                    palette=sns.color_palette(flatui), scale=0.5,
                    #   order=['Sibs','Tau','Lesion'],
                    ax=ax)
        g = sns.pointplot(x='condition', y=df_mean.iloc[:,i],hue='condition',data=df_mean, 
                    linewidth=0,
                    alpha=0.9,
                    #   order=['Sibs','Tau','Lesion'],
                    ci=None,
                    markers='d',
                    ax=ax)
        # p-value calculation
        ttest_res, pval = ttest_rel(df_mean.loc[df_mean['condition']=='1Sibs',df_mean.columns[i]],
                                    df_mean.loc[df_mean['condition']=='2Tau',df_mean.columns[i]])
        ttest_p.append(pval)
        g.legend_.remove()

plt.tight_layout()

plt.savefig(fig_dir+"conditioned bouts properties.pdf",format='PDF')
plt.show()
print(f'Sibs v.s. Tau: paired t-test p-value = \n{ttest_p}')

# %%
# 'atk_ang',
# 'pre_posture_chg',
# 'heading',
# 'pitch',
# 'pre_pitch',
# 'end_pitch',
# 'decel_rot',
# 'accel_rot',
# 'decel_ang',
# 'accel_ang',
# 'speed'
df2 = all_data_cond

y_feature = 'pre_posture_chg'
x_feature = 'heading'

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
plt.savefig(fig_dir+f"{[y_feature,x_feature]}_conditioned bouts properties.pdf",format='PDF')

# %%
# # Cumulative fractions of postures 

# defaultPlotting()
# current_palette = sns.color_palette()

# data_7S = df.loc[(df['dpf']=='7') & (df['condition']=='1Sibs'),:]
# data_7T = df.loc[(df['dpf']=='7') & (df['condition']=='2Tau'),:]

# p = sns.kdeplot(data=data_7S['pitch'],cumulative=True,color=current_palette[0],linewidth=2,label="day7_Sibs")
# p = sns.kdeplot(data=data_7T['pitch'],cumulative=True,color=current_palette[1],linewidth=2,label="day7_Tau")

# data_4S = df.loc[(df['dpf']=='4') & (df['condition']=='1Sibs'),:]
# data_4T = df.loc[(df['dpf']=='4') & (df['condition']=='2Tau'),:]

# p = sns.kdeplot(data=data_4S['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[0],label="day4_Sibs")
# p = sns.kdeplot(data=data_4T['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[1],label="day4_Sibs")

# plt.ylabel("Cumulative fractions")
# plt.xlabel("Postures with ... pitch")

# # p.set_xlim(-60,0)
# plt.show()

# # for 4 conditions X dpf:
# df = steep_data

# plt_condition = ['Sibs','Tau','Sibs','Tau']
# plt_dpf = ['4','4','7','7']
# df_absmean = pd.DataFrame()

# for i in range(4):
#     tmp = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
#     abs_mean_data = tmp.groupby('expNum')[['atk_ang','pre_posture_chg','heading','pitch']].apply(
#         lambda x: x.abs().mean()
#     )
#     abs_mean_data = abs_mean_data.assign(dpf=plt_dpf[i], condition=plt_condition[i])
#     df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)

# sns.violinplot(data=df_absmean, x='dpf',y='pitch', hue='condition',dodge=True, ci='sd')
# plt.ylabel("Deviation of posture from horizontal")

# multi_comp = MultiComparison(df_absmean['pitch'], df_absmean['dpf']+df_absmean['condition'])
# print(multi_comp.tukeyhsd().summary())
# plt.show()

# %%
# # pitch - heading. eLife 2019 Figure 1B
# g = sns.relplot(data=all_data_cond, x='pitch',y='heading',hue='condition',col='condition',row='dpf',alpha=0.1,kind='scatter')
# g.set(xlim=(-30, 30), ylim=(-90, 90))
# ax1, ax2 = g.axes[0]
# lims = [-90,90]
# for row in g.axes:
#     for ax in row:
#         ax.plot(lims,lims, ls='--',color='red')

# %%

# %%
