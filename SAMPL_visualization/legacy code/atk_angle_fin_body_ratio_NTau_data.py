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
UPDATE 211010
more bouts with resliced data. 
Therefore use ymax and ymin from sibs for 2 para fit; use ymax and ymin from lesion data for leion fit
due to the x shift in lesion data (xpos). used sibs xpos in order to get comparable sigmoid slope for lesion data
added percentage change
'''

#%%
import sys
import os,glob
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
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
import scipy

from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
# from statannot import add_stat_annotation  # pip install only. if using conda env, run <conda install pip> first

# %%
# Paste root directory here
pick_data = 'ori'

if pick_data == 'ori':
    root = "/Volumes/LabData/SAMPL_data_in_use/NefmaV3/7dd_v3"
elif pick_data == 'hets':
    root = "/Volumes/LabData/SAMPL_data_in_use/resliced/combined_7DD_hets_resliced/combined_7DD_NTau-hets_data"
elif pick_data == 'ld':
    root = "/Volumes/LabData/SAMPL_data_in_use/resliced/combined_7LD_resliced"
elif pick_data =='s':
    root = "/Volumes/LabData/SAMPL_STau_in_use"
elif pick_data == 'new':
    root = '/Volumes/LabData/SAMPL_data_in_use/SAMPL_HairCell_tmpAna'

folder_name = '{pick_data}_tmp_finBody_NTauCode'
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/SAMPL_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.mkdir(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures!')

plotKDE = 0
plotMeanData = 0
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
    # return   -1.998216 + (21.242327+1.998216)/(1 + np.exp(-(a*(x + b)))) 
    return   -3.354505 + (13.868823+3.354505)/(1 + np.exp(-(a*(x + b)))) 
    # return   c + (d-c)/(1 + np.exp(-(a*(x + b)))) 

def sigfunc_2free_lesion(x, a, b, c, d):
    # sigmoid fit for 2 free parameters. Used for getting the slope value 
    return   1.486016  + (6.571433-1.486016 )/(1 + np.exp(-(a*(x -6.289540)))) 
    # return   -3.354505 + (6.571433+3.354505)/(1 + np.exp(-(a*(x +b)))) 


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
    popt, pcov = curve_fit(sigfunc_2free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(0.1, -1, -1,1), bounds=([0,-20,-5,0],[5,20,5,50]))
    y = sigfunc_2free(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    return output_coef, output_fitted

def sigmoid_fit_lesion(df, x_range_to_fit):
    popt, pcov = curve_fit(sigfunc_2free_lesion, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(0.1, -1, -1,1), bounds=([0,-30,-5,0],[0.15,5,5,50]))
    y = sigfunc_2free_lesion(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    return output_coef, output_fitted

def sigmoid_fit_4free(df, x_range_to_fit):
    popt, pcov = curve_fit(sigfunc_4free, df['posture_chg'], df['atk_ang'],maxfev=1500,p0=(0.1, -0.2, -1,20), bounds=([0.1,-20,-5,1],[5,20,5,50]))

    y = sigfunc_4free(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma

def distribution_binned_average(df, condition):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='posture_chg')
    bins = pd.cut(df['posture_chg'], list(AVERAGE_BIN))
    grp = df.groupby(bins)
    df_out = grp[['posture_chg','atk_ang']].mean().assign(dpf=condition[0],condition=condition[4:])
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
binned_atk_angles = pd.DataFrame()
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
            
            binned_atk_angles = pd.concat([binned_atk_angles, distribution_binned_average(all_for_fit, all_conditions[condition_idx])],ignore_index=True)
            mean_data_cond = pd.concat([mean_data_cond, mean_data.assign(dpf=all_conditions[condition_idx][0],
                                                                         condition=all_conditions[condition_idx][4:])])
            all_data_cond = pd.concat([all_data_cond, all_for_fit.assign(dpf=all_conditions[condition_idx][0],
                                                                         condition=all_conditions[condition_idx][4:])])

            # jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            # for excluded_exp, idx_group in enumerate(jackknife_idx):
            #     coef, fitted_y = sigmoid_fit(all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)], X_RANGE)

            #     jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0],
            #                                                               condition=all_conditions[condition_idx][4:],
            #                                                               excluded_exp=all_for_fit.loc[all_for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            #     jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0],
            #                                                             condition=all_conditions[condition_idx][4:],
            #                                                             excluded_exp=all_for_fit.loc[all_for_fit['expNum']==excluded_exp,'date'].iloc[0])])

# %%
steep_data = all_data_cond.loc[all_data_cond['heading']>20,:]


# %% Filtering!!!! excluding attack angle < 0 & posture change > 0

# all_data_cond.drop(all_data_cond[(all_data_cond['posture_chg'] > 0) & (all_data_cond['atk_ang'] < 0)].index, inplace = True)


# rename and sort
# jackknifed_coef.columns = ['slope','locX','minY','maxY','dpf','condition','excluded_exp']
# jackknifed_coef.sort_values(by=['condition','dpf','excluded_exp'],inplace=True, ignore_index=True)            
            
all_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)            
mean_data_cond.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)   

# %%
# master fit = fit with ALL data from ALL conditions other than lesion
# to reduce free parameters, use c (min y) and d (max y) from master fit for jackknifed results
# df = all_data_cond.loc[all_data_cond['condition']=='Tau']
no_lesion_data =  all_data_cond.loc[all_data_cond['condition'].str.find('ib') !=-1,:]

df = no_lesion_data  # filter for the data you want to use for calculation
coef_master, fitted_y_master, sigma_master = sigmoid_fit_4free(df, X_RANGE)
g = sns.lineplot(x='x',y=fitted_y_master[0],data=fitted_y_master)
print(set(no_lesion_data.condition))
print(coef_master)  # a, b, c, d|
# Lesion    d   8.353048  # 7dd data for reference
# Sibs/Tau c = 1.998216 d = 21.242327  # 7dd data for reference

print('*–––––––––STOP! Read annotations––––––––*')


# %%
# previous step should print out fitted results for 4 degree of freedoms.
# locate the sigfunc_2free() function defined above, replace c and d with the coefs above
 
atk_max_lesion = 6.571433-1.486016# 6.571433-1.749411 #8.353048 # 6.571433
atk_max_other = 13.868823+3.354505 # 21.242327 [13.868823 for sibs] # also paste d value here for atk_angle calculation
# %%
jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

for condition, for_fit in all_data_cond.groupby('condition'):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    if condition.find('esion')==-1:
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y = sigmoid_fit(for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE)
            slope = coef.iloc[0,0]*atk_max_other / 4
            jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(slope=slope,
                                                                      dpf=for_fit['dpf'].iloc[0],
                                                                      condition=condition,
                                                                      excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=for_fit['dpf'].iloc[0],
                                                                    condition=condition,
                                                                    excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
    elif condition.find('esion')!=-1:
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y = sigmoid_fit_lesion(for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE)
            slope = coef.iloc[0,0]*atk_max_lesion / 4
            jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(slope=slope,
                                                                      dpf=for_fit['dpf'].iloc[0],
                                                                      condition=condition,
                                                                      excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
            jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=for_fit['dpf'].iloc[0],
                                                                    condition=condition,
                                                                    excluded_exp=for_fit.loc[for_fit['expNum']==excluded_exp,'date'].iloc[0])])
# rename and sort
jackknifed_coef.columns = ['a','locX','minY','maxY','slope','dpf','condition','excluded_exp']
jackknifed_coef.sort_values(by=['condition','dpf','excluded_exp'],inplace=True, ignore_index=True)            

# %% 
# PLOTs

# %%
# plot fitted sigmoid 
# plot code need to be personalized according to the conditions you have
hue_order = list(set(jackknifed_y.condition))
hue_order.sort()
defaultPlotting()

g = sns.lineplot(x='x',y=jackknifed_y[0],data=jackknifed_y, hue='condition',hue_order = hue_order,style='dpf',ci='sd')

# g = sns.scatterplot(x='posture_chg',y='atk_ang',hue='condition',hue_order = hue_order,s=30, data=binned_atk_angles, alpha=0.3,linewidth=1)
g = sns.lineplot(x='posture_chg',y='atk_ang',hue='condition', data=binned_atk_angles, alpha=0.3,linewidth=1)

plt.show()
# g = sns.lineplot(x='x',y=raw_y[0],data=raw_y, hue='condition',style='dpf',ci='sd')

#%%
# plot 2 paramit fit coef
f, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 8),sharex='all')
flatui = ["#D0D0D0"] * (jackknifed_coef.groupby('condition').size().max())

g1 = sns.pointplot(y='slope',x='condition', hue='excluded_exp', data=jackknifed_coef,ax=axes[0,0],
                   palette=sns.color_palette(flatui), scale=0.5,
                   )
g1 = sns.pointplot(y='slope',x='condition', hue='condition', data=jackknifed_coef,ax=axes[0,0],
                   linewidth=0,
                   alpha=0.9,
                   ci=None,
                   markers='d',)
# g1.set_yticks(np.arange(0.12,0.24,0.02))

g2 = sns.pointplot(y='locX',x='condition', hue='excluded_exp', data=jackknifed_coef,ax=axes[0,1],
                   palette=sns.color_palette(flatui), scale=0.5,                  
                   )
# g3 = sns.pointplot(y='minY',x='condition', hue='excluded_exp', data=jackknifed_coef,ax=axes[1,0],
#                    palette=sns.color_palette(flatui), scale=0.5,
#                    order=hue_order)
# g4 = sns.pointplot(y='maxY',x='condition', hue='excluded_exp', data=jackknifed_coef,ax=axes[1,1],
#                    palette=sns.color_palette(flatui), scale=0.5,
#                    order=hue_order)
g1.legend_.remove()
g2.legend_.remove()
# g3.legend_.remove()
# g4.legend_.remove()
sns.despine(trim=True)

ttest_res, ttest_p = ttest_rel(jackknifed_coef.loc[jackknifed_coef['condition']==hue_order[0],'slope'],
                               jackknifed_coef.loc[jackknifed_coef['condition']==hue_order[1],'slope'])
print(f'slope: Sibs v.s. Tau: paired t-test p-value = {ttest_p}')



ttest_res, ttest_p = scipy.stats.ttest_ind(jackknifed_coef.loc[jackknifed_coef['condition']==hue_order[0],'slope'],
                                           jackknifed_coef.loc[jackknifed_coef['condition']==hue_order[1],'slope'])
print(f'slope: Sibs v.s. Lesion: paired t-test p-value = {ttest_p}')
plt.show()


# %%
# plot slope percentage change
for string in all_conditions:  # if has lesion data or not
    if string.find("esion") != -1:
        if_plt_percentage = 1
    else:
        if_plt_percentage = 0
        
if if_plt_percentage == 1: # if contain lesion data    
    all_conditions.sort()
    tmp = jackknifed_coef.groupby('condition').mean()
    lesion_mean = tmp.iloc[-1,:]
    lesion_mean = lesion_mean['slope']
    cond0_plt =  jackknifed_coef.loc[jackknifed_coef['condition'].str.find('ib') !=-1,:]
    cond1_plt =  jackknifed_coef.loc[jackknifed_coef['condition'].str.find('au') !=-1,:]

    cond0_plt = cond0_plt.sort_values(by=['excluded_exp']).reset_index()
    cond1_plt = cond1_plt.sort_values(by=['excluded_exp']).reset_index()

    pltData_cond0 = cond0_plt.loc[:,'slope']
    pltData_cond1 = cond1_plt.loc[:,'slope']
    percentage_chg = (pltData_cond0-pltData_cond1).divide(pltData_cond0-lesion_mean) *100

    percentage_chg = 100-percentage_chg
    cond0_plt = cond0_plt.assign(y = [100] * len(percentage_chg))
    cond1_plt = cond1_plt.assign(y = percentage_chg)
    plt_data = pd.concat([cond0_plt,cond1_plt])
    
    defaultPlotting()
    
    age_condition = set(plt_data['dpf'].values)

    # initialize a multi-plot, feel free to change the plot size
    f, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.5*(1), 10), sharey='row')
    axes = axes.flatten()  # flatten if multidimenesional (multiple dpf)
    # setup color scheme for dot plots
    flatui = ["#D0D0D0"] * (plt_data.groupby('condition').size()[0])
    defaultPlotting()

    # loop through differrent age (dpf), plot parabola in the first row and sensitivy in the second.
    for i, age in enumerate(age_condition):
        
        # plot std
        std_plt = plt_data.loc[plt_data['dpf']==age]
        # plot jackknifed paired data
        p = sns.pointplot(x='condition', y='y', hue='excluded_exp',data=std_plt,
                        palette=sns.color_palette(flatui), scale=0.5,
                        ax=axes[i+1],
                    #   order=['Sibs','Tau','Lesion'],
        )
        # plot mean data
        p = sns.pointplot(x='condition', y='y',hue='condition',data=std_plt, 
                        linewidth=0,
                        alpha=0.9,
                        ci=None,
                        markers='d',
                        ax=axes[i+1],
                        #   order=['Sibs','Tau','Lesion'],
        )
        p.legend_.remove()
        # p.set_yticks(np.arange(0.1,0.52,0.04))
        p.set_ylim(0,100)
        sns.despine(trim=False)
        
        
    plt.show()
    
    
    
    
    

# %%
# Posture change - attack angle KDE joint plot

# This is a simple kde plot
# p = sns.relplot(data=all_data_cond, x='posture_chg',y='atk_ang',col='condition',row='dpf',alpha=0.1,kind='scatter')
# p.set(xlim=(-20, 20), ylim=(-20, 25))

# This is the joint plot
df = all_data_cond

plt_condition = list(set(df['condition']))
plt_condition.sort()
plt_dpf = ['7','7']

for i in range(2):
    df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
    print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
    sns.jointplot(df_to_plot['posture_chg'], df_to_plot['atk_ang'], kind="kde", height=5, space=0, xlim=(-12, 12), ylim=(-20, 25))
    # plt.show()

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
    g = sns.pointplot(x='condition',y=mean_data_cond.iloc[:,i], hue='date',data=mean_data_cond,
                  palette=sns.color_palette(flatui), scale=0.5,
                  order=['Sibs','Tau','Lesion'],
                  ax=ax)
    g = sns.pointplot(x='condition', y=mean_data_cond.iloc[:,i],hue='condition',data=mean_data_cond, 
                  linewidth=0,
                  alpha=0.9,
                  order=['Sibs','Tau','Lesion'],
                  ci=None,
                  markers='d',
                  ax=ax
                  )
    # p-value calculation
    ttest_res, ttest_p = ttest_rel(mean_data_cond.loc[mean_data_cond['condition']=='Sibs',mean_data_cond.columns[i]],
                                   mean_data_cond.loc[mean_data_cond['condition']=='Tau',mean_data_cond.columns[i]])
    print(f'{mean_data_cond.columns[i]} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

    g.legend_.remove()

plt.show()

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

# for multiple conditions under the same dpf

plt_condition = ['Sibs','Tau','Lesion']
df = steep_data
df_absmean = pd.DataFrame()
for condition in plt_condition:
    tmp = df.loc[df['condition']==condition ,:]
    abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.abs().mean()
    )
    abs_mean_data = abs_mean_data.assign(condition=condition,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)
    
p = sns.pointplot(data=df_absmean, x='condition',y='pitch', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_absmean, x='condition',y='pitch', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("Deviation of posture from horizontal")
  
ttest_res, ttest_p = ttest_rel(df_absmean.loc[df_absmean['condition']=='Sibs','pitch'],
                               df_absmean.loc[df_absmean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

# # for 4 conditions X dpf:
# df = steep_data

# plt_condition = ['Sibs','Tau','Sibs','Tau']
# plt_dpf = ['4','4','7','7']
# df_absmean = pd.DataFrame()

# for i in range(4):
#     tmp = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
#     abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
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
# Posture change (Figure 3D)

plt_condition = ['Sibs','Tau','Lesion']
df = all_data_cond
df_mean = pd.DataFrame()
for condition in plt_condition:
    tmp = df.loc[df['condition']==condition ,:]
    mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.median()
    )
    mean_data = mean_data.assign(condition=condition,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_mean = pd.concat([df_mean,mean_data],ignore_index=True)
    
p = sns.pointplot(data=df_mean, x='condition',y='pitch', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_mean, x='condition',y='pitch', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("Posture change")
  
ttest_res, ttest_p = ttest_rel(df_mean.loc[df_mean['condition']=='Sibs','pitch'],
                               df_mean.loc[df_mean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')
# %%
# # pitch - heading. eLife 2019 Figure 1B
# g = sns.relplot(data=all_data_cond, x='pitch',y='heading',hue='condition',col='condition',row='dpf',alpha=0.1,kind='scatter')
# g.set(xlim=(-30, 30), ylim=(-90, 90))
# ax1, ax2 = g.axes[0]
# lims = [-90,90]
# for row in g.axes:
#     for ax in row:
#         ax.plot(lims,lims, ls='--',color='red')

# %% ------------OTHER data -----------------

dive_data = all_data_cond.loc[all_data_cond['heading']<-10,:]
negPitch_data = all_data_cond.loc[all_data_cond['pitch']<0,:]
# negPitch_adj = all_data_cond.loc[all_data_cond['pitch']<0 & ,:]

# %%
df = negPitch_data

# Cumulative fractions of postures 

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
plt.xlabel("Postures with negative pitch")

p.set_xlim(-60,0)

plt.show()


# deviation of posture from horizontal
plt_condition = ['Sibs','Tau','Lesion']
df_absmean = pd.DataFrame()
for condition in plt_condition:
    tmp = df.loc[df['condition']==condition ,:]
    abs_mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.abs().mean()
    )
    abs_mean_data = abs_mean_data.assign(condition=condition,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_absmean = pd.concat([df_absmean,abs_mean_data],ignore_index=True)
    
p = sns.pointplot(data=df_absmean, x='condition',y='pitch', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_absmean, x='condition',y='pitch', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("Deviation of posture from horizontal")
  
ttest_res, ttest_p = ttest_rel(df_absmean.loc[df_absmean['condition']=='Sibs','pitch'],
                               df_absmean.loc[df_absmean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')


#  Posture change 

plt_condition = ['Sibs','Tau','Lesion']
df_mean = pd.DataFrame()
for condition in plt_condition:
    tmp = df.loc[df['condition']==condition ,:]
    mean_data = tmp.groupby('expNum')[['atk_ang','posture_chg','heading','pitch']].apply(
        lambda x: x.median()
    )
    mean_data = mean_data.assign(condition=condition,
                                         date=tmp.groupby('expNum')['date'].head(1).values)
    df_mean = pd.concat([df_mean,mean_data],ignore_index=True)
    
p = sns.pointplot(data=df_mean, x='condition',y='pitch', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_mean, x='condition',y='pitch', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("pitch (posture change)")
  
ttest_res, ttest_p = ttest_rel(df_mean.loc[df_mean['condition']=='Sibs','pitch'],
                               df_mean.loc[df_mean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

# atk angle
p = sns.pointplot(data=df_mean, x='condition',y='atk_ang', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_mean, x='condition',y='atk_ang', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("atk_ang")
  
ttest_res, ttest_p = ttest_rel(df_mean.loc[df_mean['condition']=='Sibs','pitch'],
                               df_mean.loc[df_mean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

# heading
p = sns.pointplot(data=df_mean, x='condition',y='heading', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_mean, x='condition',y='heading', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("heading")
  
ttest_res, ttest_p = ttest_rel(df_mean.loc[df_mean['condition']=='Sibs','pitch'],
                               df_mean.loc[df_mean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

# posture_chg
p = sns.pointplot(data=df_mean, x='condition',y='posture_chg', hue='date',                  
              palette=sns.color_palette(flatui), scale=0.5,
)
p = sns.pointplot(data=df_mean, x='condition',y='posture_chg', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
# sns.despine(trim=True)

plt.ylabel("posture_chg")
  
ttest_res, ttest_p = ttest_rel(df_mean.loc[df_mean['condition']=='Sibs','pitch'],
                               df_mean.loc[df_mean['condition']=='Tau','pitch'])
plt.show()

print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')