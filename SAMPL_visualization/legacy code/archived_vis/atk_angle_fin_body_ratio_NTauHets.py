'''
Attack angles
https://elifesciences.org/articles/45839

This script takes attack angles and pre-bout posture change and calculats the fin-body ratio using sigmoid fit
The sigmoid fit reuqires the max number of bouts. 
Please this version of the code is HARD-CODED, but the sigmoid fit requqires parameter adjustments according to your specific data anyway. 
Will make it softcoded in future versions.

This analysis only takes one variable condition, either experimental groups (sibs, tau, lesion, etc.) or age (dpf, wpf...)

DEE's originial MATLAB code can be helpful

How to calculate attack angles:
0. Plot Posture change - attack angle KDE joint plot to determine whether the distribution is good enough for sigmoid fits
1. to get the max and min for the atk_angle - posture_chg sigmoid, fit the data using 4 free parameter sigmoid fit. You may decide to combine all the conditions or exclude certain conditions.
2. get the c and d coefs, which are min and max values of the fitted sigmoid, respectivelly. You should now reach the STOP warning
3. replace the max and min values in sigfunc_2free(). Continue after the STOP warning
4. adapt the conditions in the plot code to your the conditions of your dataset.

UPDATE 210505
add filtering. excluded data with posture_change > 0 & atk_ang < 0
'''

#%%
import sys
import os,glob
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
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
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
# from statannot import add_stat_annotation  # pip install only. if using conda env, run <conda install pip> first

# %%
# Paste root directory here
root = "/Users/yunluzhu/Lab/Lab2/Data/SAMPL_data_in_use/combined_7DD_hets/combined_7DD_NTau-hets_data"

# %%
# CONSTANTS
HEADING_LIM = 90
CLIMB_MIN = 20

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
    return df_day

def sigfunc_2free(x, a, b, c, d):
    # sigmoid fit for 2 free parameters. Used for getting the slope value 
    # same equation as 4free:
    # c + (d-c)/(1 + np.exp(-(a*(x + b))))
    return   -1.998216 + (25+1.998216)/(1 + np.exp(-(a*(x + b)))) 




# def sigfunc_2free_lesion(x, a, b, c, d):
#     # sigmoid fit for 2 free parameters. Used for getting the slope value 
#     return   -1.470026 + (8.353048+1.470026)/(1 + np.exp(-(a*(x + b)))) 


def sigfunc_4free(x, a, b, c, d):
    # sigmoid fit for all 4 free parameters.
    # locY = c
    # atk_max = d
    # one_nth = 8
    # posX = b
    # A = a
    # y = locY+ (atk_max)/(1 + np.exp(-A*(x- (A * (posX) + np.log(-(locY-(one_nth-1)*atk_max)/(locY+atk_max)))/A )))
    y = c + (d-c)/(1 + np.exp(-(a*(x + b))))

    return y

def sigmoid_fit(df, x_range_to_fit):
    popt, pcov = curve_fit(sigfunc_2free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(1, -1, -1,1), bounds=([0,-20,-5,0],[2,20,0,50]))
    y = sigfunc_2free(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    return output_coef, output_fitted

# def sigmoid_fit_lesion(df, x_range_to_fit):
#     popt, pcov = curve_fit(sigfunc_2free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(1, -1, -1,1), bounds=([0,-20,-5,0],[5,20,5,50]))
#     y = sigfunc_2free_lesion(x_range_to_fit,*popt)
#     output_coef = pd.DataFrame(data=popt).transpose()
#     output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
#     return output_coef, output_fitted

def sigmoid_fit_4free(df, x_range_to_fit):
    # popt, pcov = curve_fit(sigfunc_4free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(1, -1, -1,1), bounds=([0,-20,-5,0],[5,20,5,50]))
    popt, pcov = curve_fit(sigfunc_4free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(1, -0.2, -1,20), bounds=([0.1,-20,-5,1],[5,20,0,25]))

    y = sigfunc_4free(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma
# %%
# Fit with 4 free parameters 
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
                angles = angles.assign(idx=round_half_up(len(angles)/51)*list(range(0,51)))
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
                                             'expNum':[expNum]*len(posture_chg),
                                             'date':exp[0:6]})
                all_for_fit = pd.concat([all_for_fit, for_fit], ignore_index=True)
                # sns.scatterplot(x='posture_chg',y='atk_ang',data=all_for_fit)
                
                # calculate and concat mean posture change (mean rotation) and mean of atk_ang of the current experiment
                # mean of each experiment
                mean_data = pd.concat([mean_data, pd.DataFrame(data={'atkAng':np.nanmean(atk_ang),
                                                                     'maxSpd':np.nanmean(peak_angles_day['propBoutAligned_speed']),
                                                                     'meanRot':np.nanmean(posture_chg),
                                                                     'date':exp[0:6]
                                                                     }, index=[expNum])])
                # end of exp loop
                
            mean_data_cond = pd.concat([mean_data_cond, mean_data.assign(dpf=all_conditions[condition_idx][0],
                                                                         cond1=all_conditions[condition_idx][4:])])
            all_data_cond = pd.concat([all_data_cond, all_for_fit.assign(dpf=all_conditions[condition_idx][0],cond1=all_conditions[condition_idx][4:])])

            # jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            # for excluded_exp, idx_group in enumerate(jackknife_idx):
            #     coef, fitted_y = sigmoid_fit(all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)], X_RANGE)

            #     jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0],
            #                                                               cond1=all_conditions[condition_idx][4:],
            #                                                               excluded_exp=all_for_fit.loc[all_for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            #     jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0],
            #                                                             cond1=all_conditions[condition_idx][4:],
            #                                                             excluded_exp=all_for_fit.loc[all_for_fit['expNum']==excluded_exp,'date'].iloc[0])])

steep_data = all_data_cond.loc[all_data_cond['heading']>20,:]

# %% Filtering!!!! excluding attack angle < 0 & posture change > 0

all_data_cond.drop(all_data_cond[(all_data_cond['posture_chg'] > 0) & (all_data_cond['atk_ang'] < 0)].index, inplace = True)


# rename and sort
# jackknifed_coef.columns = ['slope','locX','minY','maxY','cond0','cond1','excluded_exp']
# jackknifed_coef.sort_values(by=['cond1','cond0','excluded_exp'],inplace=True, ignore_index=True)            
            
all_data_cond.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)            
mean_data_cond.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)   

# %%
# master fit = fit with ALL data from ALL conditions other than lesion
# to reduce free parameters, use c (min y) and d (max y) from master fit for jackknifed results
df = all_data_cond.loc[all_data_cond['atk_ang']>= 0]
coef_master, fitted_y_master, sigma_master = sigmoid_fit_4free(df, X_RANGE)
g = sns.lineplot(x='x',y=fitted_y_master[0],data=fitted_y_master)

print(coef_master)  # a, b, c, d
# Lesion    d   8.353048  # 7dd data for reference
# Sibs/Tau  d   16.65836  # 7dd data for reference
# hets 0.150809  9.195434 -1.115009e-09  10.186126
print('*–––––––––STOP! Read annotations––––––––*')

# %%
# previous step should print out fitted results for 4 degree of freedoms.
# locate the sigfunc_2free() function defined above, replace c and d with the coefs above
 
# atk_max_lesion = 8.353048
atk_max_other = 27  # also paste d value here for atk_angle calculation
# %%
jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

for condition, for_fit in all_data_cond.groupby('cond1'):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    if condition != 'Lesion':
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y = sigmoid_fit(for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE)
            slope = coef.iloc[0,0]*atk_max_other / 4
            jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(slope=slope,
                                                                      dpf=for_fit['cond0'].iloc[0],
                                                                      cond1=condition,
                                                                      excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=for_fit['cond0'].iloc[0],
                                                                    cond1=condition,
                                                                    excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
    elif cond1 == 'Lesion':
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y = sigmoid_fit_lesion(for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE)
            slope = coef.iloc[0,0]*atk_max_lesion / 4
            jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(slope=slope,
                                                                      dpf=for_fit['cond0'].iloc[0],
                                                                      cond1=condition,
                                                                      excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=for_fit['cond0'].iloc[0],
                                                                    cond1=condition,
                                                                    excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
# rename and sort
jackknifed_coef.columns = ['a','locX','minY','maxY','slope','cond0','cond1','excluded_exp']
jackknifed_coef.sort_values(by=['cond1','cond0','excluded_exp'],inplace=True, ignore_index=True)            

# %% 
# PLOTs

# %%
# plot fitted sigmoid and coef
# plot code need to be personalized according to the conditions you have

# hue_order = list(set(all_data_cond['cond1']))  # use hue orders to define the order of conditions in the plots
hue_order=['Sibs','Tau']  # you may personalize the order of conditions

defaultPlotting()

g = sns.lineplot(x='x',y=jackknifed_y[0],data=jackknifed_y, hue='cond1',style='cond0',ci='sd',
                 hue_order = hue_order)
plt.show()
# g = sns.lineplot(x='x',y=raw_y[0],data=raw_y, hue='cond1',style='cond0',ci='sd')

f, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 8),sharex='all')
flatui = ["#D0D0D0"] * (jackknifed_coef.groupby('cond1').size().max())

g1 = sns.pointplot(y='slope',x='cond1', hue='excluded_exp', data=jackknifed_coef,ax=axes[0,0],
                   palette=sns.color_palette(flatui), scale=0.5,
                   order = hue_order)
g1 = sns.pointplot(y='slope',x='cond1', hue='cond1', data=jackknifed_coef,ax=axes[0,0],
                   linewidth=0,
                   alpha=0.9,
                   ci=None,
                   order=hue_order,
                   markers='d',)
# g1.set_yticks(np.arange(0.12,0.24,0.02))

g2 = sns.pointplot(y='locX',x='cond1', hue='excluded_exp', data=jackknifed_coef,ax=axes[0,1],
                   palette=sns.color_palette(flatui), scale=0.5,                  
                   order=hue_order)
g3 = sns.pointplot(y='minY',x='cond1', hue='excluded_exp', data=jackknifed_coef,ax=axes[1,0],
                   palette=sns.color_palette(flatui), scale=0.5,
                   order=hue_order)
g4 = sns.pointplot(y='maxY',x='cond1', hue='excluded_exp', data=jackknifed_coef,ax=axes[1,1],
                   palette=sns.color_palette(flatui), scale=0.5,
                   order=hue_order)
g1.legend_.remove()
g2.legend_.remove()
g3.legend_.remove()
g4.legend_.remove()
sns.despine(trim=True)

ttest_res, ttest_p = ttest_rel(jackknifed_coef.loc[jackknifed_coef['cond1']==hue_order[0],'slope'],
                               jackknifed_coef.loc[jackknifed_coef['cond1']==hue_order[1],'slope'])
print(f'slope: Sibs v.s. Tau: paired t-test p-value = {ttest_p}')



# ttest_res, ttest_p = scipy.stats.ttest_ind(jackknifed_coef.loc[jackknifed_coef['cond1']==hue_order[0],'slope'],
#                                            jackknifed_coef.loc[jackknifed_coef['cond1']==hue_order[1],'slope'])
# print(f'slope: Sibs v.s. Lesion: paired t-test p-value = {ttest_p}')
plt.show()

# %%
# Posture change - attack angle KDE joint plot

# This is a simple kde plot
# p = sns.relplot(data=all_data_cond, x='posture_chg',y='atk_ang',col='cond1',row='cond0',alpha=0.1,kind='scatter')
# p.set(xlim=(-20, 20), ylim=(-20, 25))

# This is the joint plot
df = all_data_cond

plt_cond1 = hue_order
plt_cond0 = ['7','7']

for i in range(2):
    df_to_plot = df.loc[(df['cond0']==plt_dpf[i]) & (df['cond1']==plt_condition[i]),:]
    print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
    sns.jointplot(df_to_plot['posture_chg'], df_to_plot['atk_ang'], kind="kde", height=5, space=0, xlim=(-12, 12), ylim=(-20, 25))
    # plt.show()

# %%
# plot mean attack angles, mean max speed, mean posture change (Figure 1—figure supplement 3)

# multi_comp = MultiComparison(mean_data_cond['atkAng'], mean_data_cond['cond0']+mean_data_cond['cond1'])
# print('* attack angles')
# print(multi_comp.tukeyhsd().summary())
# multi_comp = MultiComparison(mean_data_cond['maxSpd'], mean_data_cond['cond0']+mean_data_cond['cond1'])
# print('* max Speed')
# print(multi_comp.tukeyhsd().summary())
# multi_comp = MultiComparison(mean_data_cond['meanRot'], mean_data_cond['cond0']+mean_data_cond['cond1'])
# print('* mean rotation')
# print(multi_comp.tukeyhsd().summary())

defaultPlotting()
fig, axs = plt.subplots(3)
fig.set_figheight(15)
fig.set_figwidth(4)

for i, ax in enumerate(axs):
    g = sns.pointplot(x='cond1',y=mean_data_cond.iloc[:,i], hue='date',data=mean_data_cond,
                  palette=sns.color_palette(flatui), scale=0.5,
                  order=['Sibs','Tau','Lesion'],
                  ax=ax)
    g = sns.pointplot(x='cond1', y=mean_data_cond.iloc[:,i],hue='cond1',data=mean_data_cond, 
                  linewidth=0,
                  alpha=0.9,
                  order=['Sibs','Tau','Lesion'],
                  ci=None,
                  markers='d',
                  ax=ax
                  )
    # p-value calculation
    ttest_res, ttest_p = ttest_rel(mean_data_cond.loc[mean_data_cond['cond1']=='Sibs',mean_data_cond.columns[i]],
                                   mean_data_cond.loc[mean_data_cond['cond1']=='Tau',mean_data_cond.columns[i]])
    print(f'{mean_data_cond.columns[i]} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

    g.legend_.remove()

plt.show()

# %%
# Cumulative fractions of postures during climbs with trajectories greater than 20° (Figure 1E)

df = steep_data
defaultPlotting()
current_palette = sns.color_palette()

data_7S = df.loc[(df['cond0']=='7') & (df['cond1']=='Sibs'),:]
data_7T = df.loc[(df['cond0']=='7') & (df['cond1']=='Tau'),:]

p = sns.kdeplot(data=data_7S['pitch'],cumulative=True,color=current_palette[0],linewidth=2,label="day7_Sibs")
p = sns.kdeplot(data=data_7T['pitch'],cumulative=True,color=current_palette[1],linewidth=2,label="day7_Tau")

data_4S = df.loc[(df['cond0']=='4') & (df['cond1']=='Sibs'),:]
data_4T = df.loc[(df['cond0']=='4') & (df['cond1']=='Tau'),:]

p = sns.kdeplot(data=data_4S['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[0],label="day4_Sibs")
p = sns.kdeplot(data=data_4T['pitch'],cumulative=True,color=sns.color_palette("pastel", 8)[1],label="day4_Sibs")

plt.ylabel("Cumulative fractions")
plt.xlabel("Postures during climbs (deg)")

p.set_xlim(-10,90)

# %%
# deviation of posture from horizontal during steep climbs (heading > 20°) (Figure 1F)

# for multiple conditions under the same dpf

plt_cond1 = ['Sibs','Tau','Lesion']
df = steep_data
df_absmean = pd.DataFrame()
for condition in plt_condition:
    tmp = df.loc[df['cond1']==condition ,:]
    abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.abs().mean()
    )
    abs_mean_data = abs_mean_data.assign(cond1=condition,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)
    
p = sns.pointplot(data=df_absmean, x='cond1',y='pitch', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_absmean, x='cond1',y='pitch', hue='cond1',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("Deviation of posture from horizontal")

ttest_res, ttest_p = ttest_rel(df_absmean.loc[df_absmean['cond1']=='Sibs','pitch'],
                               df_absmean.loc[df_absmean['cond1']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

# # for 4 conditions X dpf:
# df = steep_data

# plt_cond1 = ['Sibs','Tau','Sibs','Tau']
# plt_cond0 = ['4','4','7','7']
# df_absmean = pd.DataFrame()

# for i in range(4):
#     tmp = df.loc[(df['cond0']==plt_dpf[i]) & (df['cond1']==plt_condition[i]),:]
#     abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
#         lambda x: x.abs().mean()
#     )
#     abs_mean_data = abs_mean_data.assign(dpf=plt_dpf[i], cond1=plt_condition[i])
#     df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)

# sns.violinplot(data=df_absmean, x='cond0',y='pitch', hue='cond1',dodge=True, ci='sd')
# plt.ylabel("Deviation of posture from horizontal")

# multi_comp = MultiComparison(df_absmean['pitch'], df_absmean['cond0']+df_absmean['cond1'])
# print(multi_comp.tukeyhsd().summary())
# plt.show()

# %%
# # pitch - heading. eLife 2019 Figure 1B
# g = sns.relplot(data=all_data_cond, x='pitch',y='heading',hue='cond1',col='cond1',row='cond0',alpha=0.1,kind='scatter')
# g.set(xlim=(-30, 30), ylim=(-90, 90))
# ax1, ax2 = g.axes[0]
# lims = [-90,90]
# for row in g.axes:
#     for ax in row:
#         ax.plot(lims,lims, ls='--',color='red')


