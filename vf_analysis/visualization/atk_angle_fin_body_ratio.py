'''
Atk angles
https://elifesciences.org/articles/45839
    root -|
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- ...
Notes 
    - a: is the dpf, only supports digits for now
    - bb: is the light-dark condition. Ignored in plotting.
    - condition: is the condition of the exp, such as control/leasion/tau... Flexible length
    - the number of folders is also flexible 
'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
# from statannot import add_stat_annotation  # pip install only. if using conda env, run <conda install pip> first

# %%
# CONSTANTS
root = "/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/test_data"
HEADING_LIM = 90
CLIMB_MIN = 20
X_RANGE = np.arange(-20,40.01,0.01)
# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="whitegrid")

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    return df_day

def sigfunc(x, a, b, c, d):
    # a: slope
    # b: posX8 = -1.11
    # c: locY = -2.9393
    # d: atk_max = 16.9779
    one_nth = 10  # constrain sigmoid to point at 1/8th height
    # return (c + (d)/(1 + np.exp(-a*(x- (a * b + np.log(-(c-(8-1)*d)/(c+d)))/a ))) )
    return   -5 + (20+5)/(1 + np.exp(-(a*(x + b)))) 
    # return d/(1 + np.exp(-a*x))

def sigmoid_fit(df, x_range_to_fit):
    # popt, pcov = curve_fit(sigfunc, df['posture_chg'], df['atk_ang'],maxfev=5000)
    popt, pcov = curve_fit(sigfunc, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(1, -1, -1,1), bounds=([0,-20,-5,0],[5,20,5,50]))
    y = sigfunc(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    return output_coef, output_fitted

# %%
# main
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)
        
jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

all_data_cond = pd.DataFrame()
mean_data_cond = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_for_fit = pd.DataFrame()
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
                posture_chg = angles.loc[peak_idx-2, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                
                for_fit = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                             'posture_chg':posture_chg.values, 
                                             'heading':peak_angles_day['propBoutAligned_heading'], 
                                             'pitch':peak_angles_day['propBoutAligned_pitch'],
                                             'expNum':[expNum]*len(posture_chg)})
                all_for_fit = pd.concat([all_for_fit, for_fit], ignore_index=True)
                # sns.scatterplot(x='posture_chg',y='atk_ang',data=all_for_fit)
                
                # calculate and concat mean posture change (mean rotation) and mean of atk_ang of the current experiment
                # mean of each experiment
                mean_data = pd.concat([mean_data, pd.DataFrame(data={'atkAng':np.nanmean(atk_ang),
                                                                     'maxSpd':np.nanmean(peak_angles_day['propBoutAligned_speed']),
                                                                     'meanRot':np.nanmean(posture_chg)
                                                                     },index=[expNum])])
                # end of exp loop
                
            mean_data_cond = pd.concat([mean_data_cond, mean_data.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
            all_data_cond = pd.concat([all_data_cond, all_for_fit.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])

            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            for idx_group in jackknife_idx:
                # coef, fitted_y = sigmoid_fit(all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)], X_RANGE)
                coef, fitted_y = sigmoid_fit(all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)], X_RANGE)

                jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
                jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])

steep_data = all_data_cond.loc[all_data_cond['heading']>20,:]

# %%
# rename and sort
jackknifed_coef.columns = ['slope','locX','minY','maxY','dpf','condition']
jackknifed_coef.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)            
            
all_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)            
mean_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)   

# %%
# master fit = fit with ALL data from ALL conditions
coef_master, fitted_y_master = sigmoid_fit(all_data_cond, X_RANGE)
g = sns.lineplot(x='x',y=fitted_y_master[0],data=fitted_y_master)

# %% 
# PLOTs


# %%
# plot fitted sigmoid and coef
defaultPlotting()

g = sns.lineplot(x='x',y=jackknifed_y[0],data=jackknifed_y, hue='condition',style='dpf',ci='sd')
plt.show()
# g = sns.lineplot(x='x',y=raw_y[0],data=raw_y, hue='condition',style='dpf',ci='sd')

f, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),sharex='all')

g1 = sns.swarmplot(y='slope',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[0,0])
g2 = sns.swarmplot(y='locX',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[0,1])
g3 = sns.swarmplot(y='minY',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[1,0])
g4 = sns.swarmplot(y='maxY',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[1,1])
plt.show()

# %%
# Posture change - attack angle KDE joint plot

# This is a simple kde plot
# p = sns.relplot(data=all_data_cond, x='posture_chg',y='atk_ang',col='condition',row='dpf',alpha=0.1,kind='scatter')
# p.set(xlim=(-20, 20), ylim=(-20, 25))

# This is the joint plot
df = all_data_cond

plt_condition = ['Sibs','Tau','Sibs','Tau']
plt_dpf = ['4','4','7','7']

for i in range(4):
    df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
    print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
    sns.jointplot(df_to_plot['posture_chg'], df_to_plot['atk_ang'], kind="kde", height=5, space=0, xlim=(-12, 12), ylim=(-20, 25))
    plt.show()

# %%
# plot mean attack angles, mean max speed, mean posture change (Figure 1—figure supplement 3)

multi_comp = MultiComparison(mean_data_cond['atkAng'], mean_data_cond['dpf']+mean_data_cond['condition'])
print('* attack angles')
print(multi_comp.tukeyhsd().summary())
multi_comp = MultiComparison(mean_data_cond['maxSpd'], mean_data_cond['dpf']+mean_data_cond['condition'])
print('* max Speed')
print(multi_comp.tukeyhsd().summary())
multi_comp = MultiComparison(mean_data_cond['meanRot'], mean_data_cond['dpf']+mean_data_cond['condition'])
print('* mean rotation')
print(multi_comp.tukeyhsd().summary())

defaultPlotting()
fig, axs = plt.subplots(3)
fig.set_figheight(15)
fig.set_figwidth(4)

for i, ax in enumerate(axs):
    sns.violinplot(x='dpf',y=mean_data_cond.iloc[:,i], hue='condition',data=mean_data_cond,scale="area", inner='box',  split=False, cut=True, ax=ax,dodge=True)
    # sns.swarmplot(x='dpf',y=mean_data_cond.iloc[:,i], hue='condition', data=mean_data_cond,palette=sns.color_palette("Blues") , dodge=True, ax=ax,size=4)

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

# %%
# deviation of posture from horizontal during steep climbs (heading > 20°) (Figure 1F)
df = steep_data

plt_condition = ['Sibs','Tau','Sibs','Tau']
plt_dpf = ['4','4','7','7']
df_absmean = pd.DataFrame()

for i in range(4):
    tmp = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
    abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.abs().mean()
    )
    abs_mean_data = abs_mean_data.assign(dpf=plt_dpf[i], condition=plt_condition[i])
    df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)

sns.violinplot(data=df_absmean, x='dpf',y='pitch', hue='condition',dodge=True, ci='sd')
plt.ylabel("Deviation of posture from horizontal")

multi_comp = MultiComparison(df_absmean['pitch'], df_absmean['dpf']+df_absmean['condition'])
print(multi_comp.tukeyhsd().summary())
plt.show()

# %%
# pitch - heading. eLife 2019 Figure 1B
g = sns.relplot(data=all_data_cond, x='pitch',y='heading',hue='condition',col='condition',row='dpf',alpha=0.1,kind='scatter')
g.set(xlim=(-30, 30), ylim=(-90, 90))
ax1, ax2 = g.axes[0]
lims = [-90,90]
for row in g.axes:
    for ax in row:
        ax.plot(lims,lims, ls='--',color='red')
