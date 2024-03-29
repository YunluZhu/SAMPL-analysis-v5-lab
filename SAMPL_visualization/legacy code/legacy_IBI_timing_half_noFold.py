'''
legacy code





This version loads "prop_Bout_IEI2" from IEI_data.h5 and reads 'propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime'
conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcde in this case)
age info is taken from the first character in folder names (x in this case, does not need to be number)
AA represents light dark conditions (LD or DD or LL...), not used.
 
outputs: 
    plots of fiitted parabola (jackknifed, half parabola), to make full parabola, change fit function to 'parabola_fit1'
    plots of fiitted coefs of function y = a * ((x-b)**2) + c (jackknifed)
    plots of paired sensitivities (jackknifed)
    paired T test results for sensitivities if number of conditions per age == 2
    multiple comparison results for sensitiivity if number of conditions per age > 2
    
UPDATE 210607: new version allows plotting of either down half (pitch < baseline pitch) or upper half (pitch > baseline pitch) of the data.
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
from numpy.polynomial import polynomial
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir

# %%
PLOT_WHICH = -1  # plot which half of the data, -1 is down, 1 is up

# Paste root directory here
pick_data = 'hc'
root = get_data_dir(pick_data)

folder_name = '{pick_data}_tmp_SensitivityHalf' + str(PLOT_WHICH)
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/SAMPL_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.mkdir(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('fig folder already exist')

MEAN_X_INTERSECT = 4
# MEAN_X_INTERSECT = 3.919342  # for YZ data
# mean coef of 7dpf
    # sensitivity    3.513709e-04
    # x_inter        2.215397e+00
    # y_inter        7.581419e-01
    # dpf            5.354752e+80
    
# %%
# CONSTANTS
SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 4 

if PLOT_WHICH == -1:
    X_RANGE = range(-81,0,1)
elif PLOT_WHICH == 1:
    X_RANGE = range(0,81,1)
else:
    print("Invalide PLOT_WHICH. Tell me which part of the pitch data to plot")
    sys.exit()
                
X_RANGE_FULL = range(-80,81,1)

# %%
def defaultPlotting():
    '''plot style'''
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def day_night_split2(df,time_col_name):
    '''extra day data only (9AM to 11PM)'''
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    return df_day

# def even_binned_average(df, samples_per_bin, condition):
#     '''this function works similar to the makeEvenHistogram.m by DEE
#     bins raw pitch data and return mean. Used to plot data points on the parabola.'''
#     df = df.sort_values(by='propBoutIEI_pitch')
#     df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
#     df_out = df.groupby(np.arange(len(df))//samples_per_bin)[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0:2],cond1=condition[4:])
#     return df_out

def distribution_binned_average(df, xBins, condition):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], xBins)
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0:2],cond1=condition[4:])
    return df_out 
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def ffunc2(x, a, c):
    return a*((x-MEAN_X_INTERSECT)**2)+c

def parabola_fit1(df, X_RANGE_to_fit = X_RANGE_FULL):
    # fit bout probability - pitch to parabola
    # May need to adjust bounds
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['y_boutFreq'], p0=(0.0002,3,0.8) , maxfev=1500,bounds=((0, 0, 0),(0.5, 15, 1)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(cond1=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc2(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

def parabola_fit_centered(df, X_RANGE_to_fit):
    '''
    Fit bout probability - pitch to parabola
    Because data points are mostly symmetric, fold along x=0.
    '''    
    # May need to adjust bounds
    popt, pcov = curve_fit(ffunc2, df['propBoutIEI_pitch'], df['y_boutFreq'], p0=(0.0002,0.64) ,maxfev=1500, bounds=((0, 0.63),(2, 0.65)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(cond1=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc2(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %%
# Main function
# get the name of all folders under root
all_conditions = []
folder_paths = []
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

# initialize results dataframe
all_cond_angles = pd.DataFrame()  # all ori pitch including all conditions, for validation only, not needed for plotting
binned_angles = pd.DataFrame()  # binned mean of pitch for plotting as "raw data"
coef_ori = pd.DataFrame()  # coef results calculated with all original pitch data
fitted_y_ori = pd.DataFrame()  # fitted y using all original pitch data
jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

# for each folder (condition)
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_day_angles = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')               
                body_angles = df.loc[:,['propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime']]
                day_angles = day_night_split2(body_angles,'propBoutIEItime').assign(expNum=expNum, date=exp[0:6])
                day_angles.dropna(inplace=True)
                all_day_angles = pd.concat([all_day_angles, day_angles[['propBoutIEI', 'propBoutIEI_pitch','expNum','date']]],ignore_index=True)
                # Enter next folder under this condition
                
            all_day_angles = all_day_angles.assign(y_boutFreq=1/all_day_angles['propBoutIEI'])
            # get all angles at all conditions, for validation. not needed for plotting
            all_cond_angles = pd.concat([all_cond_angles,all_day_angles.assign(cond1=all_conditions[condition_idx])],ignore_index=True)
                        
            # # code for flipping negative data
            # all_day_angles['propBoutIEI_pitch'] = (all_day_angles['propBoutIEI_pitch'] - MEAN_X_INTERSECT).abs()

            # choose which part of the pitch data to plot, determined by PLOT_WHICH == -1 or 1
            if PLOT_WHICH == -1: # plot down
                all_day_angles = all_day_angles.loc[all_day_angles.propBoutIEI_pitch <= MEAN_X_INTERSECT]
                
            elif PLOT_WHICH == 1: # plot up
                all_day_angles = all_day_angles.loc[all_day_angles.propBoutIEI_pitch >= MEAN_X_INTERSECT]
            
            # get binned mean of angles for plotting "raw" data 
            if PLOT_WHICH == -1:
                xBins = list(np.arange(-90,0,BIN_WIDTH))
            elif PLOT_WHICH == 1:
                xBins = list(np.arange(0,90,BIN_WIDTH))  
                
            binned_angles = pd.concat([binned_angles, distribution_binned_average(all_day_angles, xBins, all_conditions[condition_idx])],ignore_index=True)
            
            # fit angles condition by condition and concatenate results
            coef, fitted_y = parabola_fit_centered(all_day_angles, X_RANGE)
            coef_ori = pd.concat([coef_ori, coef.assign(dpf=all_conditions[condition_idx][0:2],cond1=all_conditions[condition_idx][4:])])
            fitted_y_ori = pd.concat([fitted_y_ori, fitted_y.assign(dpf=all_conditions[condition_idx][0:2],cond1=all_conditions[condition_idx][4:])])
            
            # jackknife for the index
            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                coef, fitted_y = parabola_fit_centered(all_day_angles.loc[all_day_angles['expNum'].isin(idx_group)], X_RANGE)
                jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0:2],
                                                                        cond1=all_conditions[condition_idx][4:],
                                                                        excluded_exp=all_day_angles.loc[all_day_angles['expNum']==excluded_exp,'date'].iloc[0])])
                jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0:2],
                                                                        cond1=all_conditions[condition_idx][4:],
                                                                        excluded_exp=all_day_angles.loc[all_day_angles['expNum']==excluded_exp,'date'].iloc[0])])
            # Enter next condition

jackknifed_coef.columns = ['sensitivity','y_inter','cond0','cond1','jackknife_excluded_sample']
jackknifed_coef.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)
jackknifed_coef['sensitivity'] = jackknifed_coef['sensitivity']*1000  # unit: mHz/deg**2

jackknifed_y.columns = ['y','x','cond0','cond1','jackknife_excluded_sample']
jackknifed_y.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)
binned_angles.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)

coef_ori.columns = ['sensitivity','y_inter','cond0','cond1']
coef_ori.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True)

fitted_y_ori.columns = ['y','x','cond0','cond1']         
fitted_y_ori.sort_values(by=['cond1','cond0'],inplace=True, ignore_index=True) 

# %%
# plot fitted parabola and sensitivity

defaultPlotting()

# Separate data by age.
age_cond1 = set(jackknifed_y['cond0'].values)
age_cond_num = len(age_condition)

# initialize a multi-plot, feel free to change the plot size
f, axes = plt.subplots(nrows=2, ncols=age_cond_num, figsize=(2.5*(age_cond_num), 10), sharey='row')
axes = axes.flatten()  # flatten if multidimenesional (multiple dpf)
# setup color scheme for dot plots
flatui = ["#D0D0D0"] * (jackknifed_coef.groupby('cond1').size()[0])
defaultPlotting()

# loop through differrent age (dpf), plot parabola in the first row and sensitivy in the second.
for i, age in enumerate(age_condition):
    fitted = jackknifed_y.loc[jackknifed_y['cond0']==age]
    # dots are plotted with binned average pitches
    binned = binned_angles.loc[binned_angles['cond0']==age]
    g = sns.lineplot(x='x',y='y',hue='cond1',data=fitted, ci="sd", ax=axes[i])
    g = sns.scatterplot(x='propBoutIEI_pitch',y='y_boutFreq',hue='cond1',s=30, data=binned, alpha=0.3, ax=axes[i],linewidth=0)
    # adjust ticks
    if PLOT_WHICH == -1:
        g.set_xticks(np.arange(0,-120,-30))
    elif PLOT_WHICH ==1:
        g.set_xticks(np.arange(0,120,30))

    g.set_ylim(0, None,30)
    g.set_xlim(min(X_RANGE), max(X_RANGE))
    # plot sensitivity
    coef_plt = jackknifed_coef.loc[jackknifed_coef['cond0']==age]
    # plot jackknifed paired data
    p = sns.pointplot(x='cond1', y='sensitivity', hue='jackknife_excluded_sample',data=coef_plt,
                    palette=sns.color_palette(flatui), scale=0.5,
                    ax=axes[i+age_cond_num],
                #   order=['Sibs','Tau','Lesion'],
    )
    # plot mean data
    p = sns.pointplot(x='cond1', y='sensitivity',hue='cond1',data=coef_plt, 
                    linewidth=0,
                    alpha=0.9,
                    ci=None,
                    markers='d',
                    ax=axes[i+age_cond_num],
                    #   order=['Sibs','Tau','Lesion'],
    )
    p.legend_.remove()
    g.set_xlim(min(X_RANGE), max(X_RANGE))
    # p.set_yticks(np.arange(0.1,0.52,0.04))
    
    # sns.despine(trim=True)
        
    condition_s = set(coef_plt['cond1'].values)
    condition_s = list(condition_s)
    
    # Paired T Test for 2 conditions 
    if len(condition_s) == 2:      
        # Separate data by condition.
        coef_cond1 = coef_plt.loc[coef_plt['cond1']==condition_s[0]].sort_values(by='jackknife_excluded_sample')
        coef_cond1 = coef_plt.loc[coef_plt['cond1']==condition_s[1]].sort_values(by='jackknife_excluded_sample')
        ttest_res, ttest_p = ttest_rel(coef_cond1['sensitivity'],coef_cond1['sensitivity'])
        print(f'* Age {age}: {condition_s[0]} v.s. {condition_s[1]} paired t-test p-value = {ttest_p}')
    elif len(condition_s) > 2:      
        multi_comp = MultiComparison(coef_plt['sensitivity'], coef_plt['cond0']+coef_plt['cond1'])
        print(f'* Age {age}:' )
        print(multi_comp.tukeyhsd().summary())
    else:
        pass
plt.show()

# %%
# some other versions of plots, may need to change hard-coded condition names and dpf

# # paired pointplot
# defaultPlotting()

# plt.figure(figsize=(2.5,7))

# df = jackknifed_coef.loc[jackknifed_coef['cond1'] != 'Lesion']
# flatui = ["#D0D0D0"] * (df.groupby('cond1').size()[0])

# p = sns.pointplot(x='cond1', y='sensitivity', hue='jackknife_excluded_sample',data=jackknifed_coef,
#                   palette=sns.color_palette(flatui), scale=0.5,
#                 #   order=['Sibs','Tau','Lesion'],
# )
# p = sns.pointplot(x='cond1', y='sensitivity',hue='cond1',data=jackknifed_coef, 
#                   linewidth=0,
#                   alpha=0.9,
#                 #   order=['Sibs','Tau','Lesion'],
#                   ci=None,
#                   markers='d'
#                   )
# p.legend_.remove()
# # p.set_yticks(np.arange(0.1,0.52,0.04))
# sns.despine(trim=True)
# print(f'Sibs v.s. Tau: paired t-test p-value = {ttest_p}')
# plt.show()

# # swarm plot 
# defaultPlotting()

# f, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12),sharex='all')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.4, hspace=None)
# g1 = sns.swarmplot(y='sensitivity',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[0])
# g1.set_ylim([0,0.001])
# g2 = sns.swarmplot(y='x_inter',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[1])
# # g2.set_ylim([4.999,5.001])
# sns.swarmplot(y='y_inter',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[2])

# # %%
# # boxplot
# defaultPlotting()

# f, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 16),sharex='all')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.4, hspace=0.1)
# g1 = sns.boxplot(y='sensitivity',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[0])
# g1.set_ylim([0,0.001])
# g2 = sns.boxplot(y='x_inter',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[1])
# # g2.set_ylim([4.999,5.001])
# sns.boxplot(y='y_inter',x='cond0', hue='cond1', data=jackknifed_coef,ax=axes[2])
