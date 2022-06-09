'''
This version uses "propBoutIEIAligned_pitch" column in the "prop_bout_IEI_aligned" dataframe from IEI_data.h5 file
which includes body angles at all frames between qualified bouts
Output: plots of distribution of body angles across different conditioins

paired T test results for standard deviation of posture  if number of conditions per age == 2
multiple comparison results for standard deviation of posture if number of conditions per age > 2

Conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcde in this case)
Age info is taken from the first character in folder names (x in this case, does not need to be number)
AA represents light dark conditions (LD or DD or LL...), not used.
'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
# %%
# Paste root directory here
pick_data = 'hc4'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data}_pitchSTD'
folder_dir = f'/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/{pick_data}'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('fig folder already exist')

DayNight_select = 'day'

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

# %%
# Main function
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

bins = list(range(-90,95,5))

jack_y_all = pd.DataFrame()
ang_std_all = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_angles = pd.DataFrame()
            exp_date_match = pd.DataFrame()
            ang_std = []
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI_aligned')               
                # get pitch
                body_angles = df.loc[:,['propBoutIEIAligned_pitch']].rename(columns={'propBoutIEIAligned_pitch':f'exp{expNum}'}).transpose()
                all_angles = pd.concat([all_angles, body_angles])
                exp_date_match = pd.concat([exp_date_match, pd.DataFrame( data= {'expNum':expNum,'date':[exp[0:6]]} )],ignore_index=True)

            cond1 = all_conditions[condition_idx].split("_")[0]
            cond2 = all_conditions[condition_idx].split("_")[1]
            # jackknife for the index
            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            # get the distribution of every jackknifed sample
            jack_y = pd.concat([pd.DataFrame(np.histogram(all_angles.iloc[idx_group].to_numpy().flatten(), bins=bins, density=True)) for idx_group in jackknife_idx], axis=1).transpose()
            jack_y_all = pd.concat([jack_y_all, jack_y.assign(age=cond1, condition=cond2)], axis=0, ignore_index=True)
            # get the std of every jackknifed sample
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                ang_std.append(np.nanstd(all_angles.iloc[idx_group].to_numpy().flatten())) 
            ang_std = pd.DataFrame(ang_std).assign(excluded_exp=exp_date_match['date'])
            ang_std_all = pd.concat([ang_std_all, ang_std.assign(age=cond1, condition=cond2)], axis=0, ignore_index=True)

jack_y_all.columns = ['Probability','Posture (deg)','dpf','condition']
jack_y_all.sort_values(by=['condition'],inplace=True)
ang_std_all.columns = ['std(posture)','excluded_exp','dpf','condition']                
ang_std_all.sort_values(by=['condition'],inplace=True)

# %%
# Plot posture distribution and its standard deviation

defaultPlotting()

# Separate data by age.
age_condition = list(jack_y_all['dpf'].values)
age_condition.sort()
age_condition = set(age_condition)
age_cond_num = len(age_condition)

# initialize a multi-plot, feel free to change the plot size
f, axes = plt.subplots(nrows=2, ncols=age_cond_num, figsize=(2.5*(age_cond_num), 10), sharey='row')
axes = axes.flatten()  # flatten if multidimenesional (multiple dpf)
# setup color scheme for dot plots
flatui = ["#D0D0D0"] * (ang_std_all.groupby('condition').size()[0])
defaultPlotting()

# loop through differrent age (dpf), plot parabola in the first row and sensitivy in the second.
for i, age in enumerate(age_condition):
    fitted = jack_y_all.loc[jack_y_all['dpf']==age]
    g = sns.lineplot(x='Posture (deg)',y='Probability', hue='condition', style='dpf', data=fitted, ci='sd', err_style='band', ax=axes[i])
    # g.set_yticks(np.arange(x,y,step))  # adjust y ticks
    g.set_xticks(np.arange(-90,135,45))  # adjust x ticks
    g.set(title=age)

    # plot std
    std_plt = ang_std_all.loc[ang_std_all['dpf']==age]
    # plot jackknifed paired data
    p = sns.pointplot(x='condition', y='std(posture)', hue='excluded_exp',data=std_plt,
                    palette=sns.color_palette(flatui), scale=0.5,
                    ax=axes[i+age_cond_num],
                #   order=['Sibs','Tau','Lesion'],
    )
    # plot mean data
    p = sns.pointplot(x='condition', y='std(posture)',hue='condition',data=std_plt, 
                    linewidth=0,
                    alpha=0.9,
                    ci=None,
                    markers='d',
                    ax=axes[i+age_cond_num],
                    #   order=['Sibs','Tau','Lesion'],
    )
    p.legend_.remove()
    # p.set_yticks(np.arange(0.1,0.52,0.04))
    sns.despine(trim=False)
    
    condition_s = set(std_plt['condition'].values)
    condition_s = list(condition_s)

    if len(condition_s) == 2:      
        # Paired T Test for 2 conditions
        # Separate data by condition.
        std_cond1 = std_plt.loc[std_plt['condition']==condition_s[0]].sort_values(by='excluded_exp')
        std_cond2 = std_plt.loc[std_plt['condition']==condition_s[1]].sort_values(by='excluded_exp')
        ttest_res, ttest_p = ttest_rel(std_cond1['std(posture)'],std_cond2['std(posture)'])
        print(f'* Age {age}: {condition_s[0]} v.s. {condition_s[1]} paired t-test p-value = {ttest_p}')
    elif len(condition_s) > 2: 
        multi_comp = MultiComparison(ang_std_all['std(posture)'], ang_std_all['dpf']+ang_std_all['condition'])
        print(f'* Age {age}:' )
        print(multi_comp.tukeyhsd().summary())
    else:
        pass
plt.savefig(fig_dir+f"/{pick_data}'s IEIpitchAligned.pdf",format='PDF')


