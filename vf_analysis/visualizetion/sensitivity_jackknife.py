'''
This version loads "prop_Bout_IEI2" from IEI_data.h5 and reads 'propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime'
outputs: 
    plots of fiitted parabola (jackknifed)
    plots of fiitted coefs of function y = a * ((x-b)**2) + c (jackknifed)
    Multiple comparison results of sensitiivity, x/y intercepts across differenet conditions (turned off)

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
from numpy.polynomial import polynomial
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# %%
# CONSTANTS
root = "/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/test_data"
SAMPLES_PER_BIN = 70
x_range = range(-80,80,1)
# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="whitegrid")

def day_night_split(df):
    hour = df['propBoutIEItime'].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, ['propBoutIEI', 'propBoutIEI_pitch']]
    return df_day


def day_night_split2(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    return df_day


def makeEvenHistogram(df, samples_per_bin, condition):
    df = all_day_angles.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    df_out = df.groupby(np.arange(len(df))//samples_per_bin)[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0],condition=condition[4:])
    return df_out

def ffunc(x, a, b, c):
    return a*((x-b)**2)+c

def parabola_fit2(df, x_range_to_fit):
    popt, pcov = curve_fit(ffunc, df['propBoutIEI_pitch'], df['y_boutFreq'], p0=(0.0002,3,0.8) , bounds=((0, 0, 0),(0.5, 15, 1)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = ffunc(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    return output_coef, output_fitted
    
# %%
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
            ang_std = []
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')               
                body_angles = df.loc[:,['propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime']]
                day_angles = day_night_split2(body_angles,'propBoutIEItime').assign(expNum=expNum)
                day_angles.dropna(inplace=True)
                all_day_angles = pd.concat([all_day_angles, day_angles[['propBoutIEI', 'propBoutIEI_pitch','expNum']]],ignore_index=True)
                # # centralize pitch angles around 0 deg and reflect the negative side. Turn off for more robust results
                # all_day_angles['propBoutIEI_pitch'] = (all_day_angles['propBoutIEI_pitch'] - all_day_angles['propBoutIEI_pitch'].mean()).abs()
                all_day_angles = all_day_angles.assign(y_boutFreq=1/all_day_angles['propBoutIEI'])
            
            # # get all angles at all conditions, for validation. not needed for plotting
            # all_cond_angles = pd.concat([all_cond_angles,all_day_angles.assign(condition=all_conditions[condition_idx])],ignore_index=True)
            
            # get binned mean of angles for plotting "raw" data 
            binned_angles = pd.concat([binned_angles, makeEvenHistogram(all_day_angles, SAMPLES_PER_BIN, all_conditions[condition_idx])],ignore_index=True)
            
            # fit angles condition by condition and concatenate results
            coef, fitted_y = parabola_fit2(all_day_angles, x_range)
            coef_ori = pd.concat([coef_ori, coef.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
            fitted_y_ori = pd.concat([fitted_y_ori, fitted_y.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
            
            # jackknife for the index
            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            for idx_group in jackknife_idx:
                coef, fitted_y = parabola_fit2(all_day_angles.loc[all_day_angles['expNum'].isin(idx_group)], x_range)
                jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
                jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])

jackknifed_coef.columns = ['sensitivity','x_inter','y_inter','dpf','condition']
jackknifed_coef.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)

jackknifed_y.columns = ['y','x','dpf','condition']
jackknifed_y.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)

coef_ori.columns = ['sensitivity','x_inter','y_inter','dpf','condition']
coef_ori.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)

fitted_y_ori.columns = ['y','x','dpf','condition']         
fitted_y_ori.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True) 

# %%
# Statistics. Multiple comparison. Uncomment if want to print the results

multi_comp = MultiComparison(jackknifed_coef['sensitivity'], jackknifed_coef['dpf']+jackknifed_coef['condition'])
# print(multi_comp.tukeyhsd().summary())
multi_comp = MultiComparison(jackknifed_coef['x_inter'], jackknifed_coef['dpf']+jackknifed_coef['condition'])
# print(multi_comp.tukeyhsd().summary())
multi_comp = MultiComparison(jackknifed_coef['y_inter'], jackknifed_coef['dpf']+jackknifed_coef['condition'])
# print(multi_comp.tukeyhsd().summary())

# %%
# plot 

# 1. parabola
# fitted lines are plotted with jackknifed data ± sd
# dots are plotted with binned average pitches

fitted7 = jackknifed_y.loc[jackknifed_y['dpf']=='7']
fitted4 = jackknifed_y.loc[jackknifed_y['dpf']=='4']
binned7 = binned_angles.loc[binned_angles['dpf']=='7']
binned4 = binned_angles.loc[binned_angles['dpf']=='4']

defaultPlotting()

f, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6),sharey='all')

g = sns.lineplot(x='x',y='y',hue='condition',data=fitted4, ci="sd", ax=axes[0])
g = sns.scatterplot(x='propBoutIEI_pitch',y='y_boutFreq',hue='condition',s=30, data=binned4, alpha=0.5, ax=axes[0])

g3 = sns.lineplot(x='x',y='y',hue='condition',data=fitted7, ci="sd", ax=axes[1])
g4 = sns.scatterplot(x='propBoutIEI_pitch',y='y_boutFreq',hue='condition',s=30, data=binned7, alpha=0.5, ax=axes[1])
plt.show()

# 2. Violin plot of jackknifed coef

defaultPlotting()

f, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12),sharex='all')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.1)
g1 = sns.violinplot(y='sensitivity',x='dpf', hue='condition',scale='area',cut=True,data=jackknifed_coef,ax=axes[0])
g1.set_ylim([0,0.0008])
g2 = sns.violinplot(y='x_inter',x='dpf', hue='condition',scale='area',cut=True, data=jackknifed_coef,ax=axes[1])
# g2.set_ylim([4.999,5.001])
sns.violinplot(y='y_inter',x='dpf', hue='condition', scale='area',cut=True,data=jackknifed_coef,ax=axes[2])
plt.show()


# %%
# some other versions of plots

# # swarm plot 
# defaultPlotting()

# f, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12),sharex='all')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.4, hspace=None)
# g1 = sns.swarmplot(y='sensitivity',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[0])
# g1.set_ylim([0,0.001])
# g2 = sns.swarmplot(y='x_inter',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[1])
# # g2.set_ylim([4.999,5.001])
# sns.swarmplot(y='y_inter',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[2])

# # %%
# # boxplot
# defaultPlotting()

# f, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 16),sharex='all')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.4, hspace=0.1)
# g1 = sns.boxplot(y='sensitivity',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[0])
# g1.set_ylim([0,0.001])
# g2 = sns.boxplot(y='x_inter',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[1])
# # g2.set_ylim([4.999,5.001])
# sns.boxplot(y='y_inter',x='dpf', hue='condition', data=jackknifed_coef,ax=axes[2])
