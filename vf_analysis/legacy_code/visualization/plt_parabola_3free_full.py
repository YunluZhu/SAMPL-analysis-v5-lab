'''
This version loads "prop_Bout_IEI2" from IEI_data.h5 and reads 'propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime'
conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcde in this case)
age info is taken from the first character in folder names (x in this case, does not need to be number)
AA represents light dark conditions (LD or DD or LL...), not used.
 
outputs: 
    plots of fiitted parabola (jackknifed, FULL parabola), to make half parabola, run script parabola_sensitivity_half.py
    plots of fiitted coefs of function y = a * ((x-b)**2) + c (jackknifed)
    plots of paired sensitivities (jackknifed)
    paired T test results for sensitivities if number of conditions per age == 2
    multiple comparison results for sensitiivity if number of conditions per age > 2

NOTE: bounds in parabola_fit1() confines the upper and lower limites of the parameters. These may need to be changed according to your own data sets.
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
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# %%
# Paste root directory here
root = "/Users/yunluzhu/Lab/Lab2/Data/VF/vf_data/combined_TTau_data"

# %%
# CONSTANTS
# SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
X_RANGE_FULL = range(-80,81,1)
# mean coef of 7dpf
    # sensitivity    3.513709e-04
    # x_inter        2.215397e+00
    # y_inter        7.581419e-01
    
# %%
def defaultPlotting():
    '''plot style'''
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def day_night_split2(df,time_col_name):
    '''extra day data only (9AM to 11PM)'''
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    return df_day

# def even_binned_average(df, samples_per_bin, condition):
#     '''this function works similar to the makeEvenHistogram.m by DEE
#     bins raw pitch data and return mean. Used to plot data points on the parabola.'''
#     df = df.sort_values(by='propBoutIEI_pitch')
#     df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
#     df_out = df.groupby(np.arange(len(df))//samples_per_bin)[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0],condition=condition[4:])
#     return df_out

def distribution_binned_average(df, bin_width, condition):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0],condition=condition[4:])
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit = X_RANGE_FULL):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['y_boutFreq'], p0=(0.0002,3,0.8) , bounds=((0, -10, 0),(0.5, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %%
# Main function
def sensitivity_jacknife_fullP(root):
    # get the name of all folders under root
    all_conditions = []
    folder_paths = []
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    # initialize results dataframe
    all_cond_angles = pd.DataFrame()  # all ori pitch including all conditions, for validation 
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
                    # enter next folder under this condition
                    
                all_day_angles = all_day_angles.assign(y_boutFreq=1/all_day_angles['propBoutIEI'])

                # get binned mean of angles for plotting "raw" data 
                binned_angles = pd.concat([binned_angles, distribution_binned_average(all_day_angles, BIN_WIDTH, all_conditions[condition_idx])],ignore_index=True)
                
                # fit angles condition by condition and concatenate results
                coef, fitted_y = parabola_fit1(all_day_angles, X_RANGE_FULL)
                coef_ori = pd.concat([coef_ori, coef.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
                fitted_y_ori = pd.concat([fitted_y_ori, fitted_y.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
                
                # jackknife for the index
                jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
                for excluded_exp, idx_group in enumerate(jackknife_idx):
                    coef, fitted_y = parabola_fit1(all_day_angles.loc[all_day_angles['expNum'].isin(idx_group)], X_RANGE_FULL)
                    jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=all_conditions[condition_idx][0],
                                                                            condition=all_conditions[condition_idx][4:],
                                                                            excluded_exp=all_day_angles.loc[all_day_angles['expNum']==excluded_exp,'date'].iloc[0])])
                    jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=all_conditions[condition_idx][0],
                                                                            condition=all_conditions[condition_idx][4:],
                                                                            excluded_exp=all_day_angles.loc[all_day_angles['expNum']==excluded_exp,'date'].iloc[0])])
        
                # get all angles at all conditions, for validation. not needed for plotting
                all_cond_angles = pd.concat([all_cond_angles,all_day_angles.assign(condition=all_conditions[condition_idx])],ignore_index=True)
                # enter next condition

    coef_all_cond, fitted_y_all_cond = parabola_fit1(all_cond_angles, X_RANGE_FULL)


    jackknifed_coef.columns = ['sensitivity','x_inter','y_inter','dpf','condition','jackknife_excluded_sample']
    jackknifed_coef.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)
    jackknifed_coef['sensitivity'] = jackknifed_coef['sensitivity']*1000  # unit: mHz/deg**2

    jackknifed_y.columns = ['y','x','dpf','condition','jackknife_excluded_sample']
    jackknifed_y.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)
    binned_angles.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)

    coef_ori.columns = ['sensitivity','x_inter','y_inter','dpf','condition']
    coef_ori.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True)

    fitted_y_ori.columns = ['y','x','dpf','condition']         
    fitted_y_ori.sort_values(by=['condition','dpf'],inplace=True, ignore_index=True) 

    # %%
    print("Fitted coefs using ALL data (for reference):")
    print(coef_all_cond)

    # plot fitted parabola and sensitivity
    defaultPlotting()

    # Separate data by age.
    age_condition = set(jackknifed_y['dpf'].values)
    age_cond_num = len(age_condition)

    # initialize a multi-plot, feel free to change the plot size
    f, axes = plt.subplots(nrows=4, ncols=age_cond_num, figsize=(2.5*(age_cond_num), 12), sharey='row')
    axes = axes.flatten()  # flatten if multidimenesional (multiple dpf)
    # setup color scheme for dot plots
    flatui = ["#D0D0D0"] * (jackknifed_coef.groupby('condition').size()[0])
    defaultPlotting()

    # loop through differrent age (dpf), plot parabola in the first row and sensitivy in the second.
    for i, age in enumerate(age_condition):
        fitted = jackknifed_y.loc[jackknifed_y['dpf']==age]
        # dots are plotted with binned average pitches
        binned = binned_angles.loc[binned_angles['dpf']==age]
        g = sns.lineplot(x='x',y='y',hue='condition',data=fitted, ci="sd", ax=axes[i])
        g = sns.scatterplot(x='propBoutIEI_pitch',y='y_boutFreq',hue='condition',s=20, data=binned, alpha=0.3, ax=axes[i],linewidth=0)
        g.set_xticks(np.arange(-90,135,45))      # adjust ticks
        g.set_ylim(0, None,30)
        
        # SENSITIVITY
        coef_plt = jackknifed_coef.loc[jackknifed_coef['dpf']==age]
        # plot jackknifed paired data
        p = sns.pointplot(x='condition', y='sensitivity', hue='jackknife_excluded_sample',data=coef_plt,
                        palette=sns.color_palette(flatui), scale=0.5,
                        ax=axes[i+age_cond_num],
                    #   order=['Sibs','Tau','Lesion'],
        )
        # plot mean data
        p = sns.pointplot(x='condition', y='sensitivity',hue='condition',data=coef_plt, 
                        linewidth=0,
                        alpha=0.9,
                        ci=None,
                        markers='d',
                        ax=axes[i+age_cond_num],
                        #   order=['Sibs','Tau','Lesion'],
        )
        p.legend_.remove()
        # p.set_yticks(np.arange(0.1,0.52,0.04))
        
        # sns.despine(trim=True)
        
        # p values for sensitivity
        condition_s = set(coef_plt['condition'].values)
        condition_s = list(condition_s)
        
        # Paired T Test for 2 conditions 
        if len(condition_s) == 2:      
            # Separate data by condition.
            coef_cond1 = coef_plt.loc[coef_plt['condition']==condition_s[0]].sort_values(by='jackknife_excluded_sample')
            coef_cond2 = coef_plt.loc[coef_plt['condition']==condition_s[1]].sort_values(by='jackknife_excluded_sample')
            ttest_res, ttest_p = ttest_rel(coef_cond1['sensitivity'],coef_cond2['sensitivity'])
            print(f'* Age {age} Sensitivity: {condition_s[0]} v.s. {condition_s[1]} paired t-test p-value = {ttest_p}')
        elif len(condition_s) > 2:      
            multi_comp = MultiComparison(coef_plt['sensitivity'], coef_plt['dpf']+coef_plt['condition'])
            print(f'* Age {age} Sensitivity:' )
            print(multi_comp.tukeyhsd().summary())
        else:
            pass
        
        # X INTERSECT
        sns.swarmplot(x='condition', y='x_inter', data=coef_plt,ax=axes[i+age_cond_num*2])
        # Y INTERSECT
        sns.swarmplot(x='condition', y='y_inter', data=coef_plt,ax=axes[i+age_cond_num*3])
        
    plt.show()
