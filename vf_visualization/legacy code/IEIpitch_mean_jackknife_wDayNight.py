'''
This version uses "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" file
which includes mean of body angles during IEI
Output: plots of distribution of body angles across different conditioins

paired T test results for standard deviation of posture  if number of conditions per age == 2
multiple comparison results for standard deviation of posture if number of conditions per age > 2

Conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcde in this case)
age info is taken from the first character in folder names (x in this case, does not need to be number)
AA represents light dark conditions (LD or DD or LL...), not used.
'''

#%%
import sys
import os,glob
import time
from turtle import end_fill
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir
from plot_functions.plt_tools import (day_night_split,defaultPlotting,set_font_type)

set_font_type()
# %%
# Paste root directory here
pick_data = 'hc4'
ztime = 'all'

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data}_pitchSTD'
folder_dir = f'/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/{pick_data}_z{ztime}'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
# main function
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
df_IBIangles = pd.DataFrame()
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
                df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')
                df_ztimed,day_index,night_index = day_night_split(df,'propBoutIEItime',ztime)
                # get pitch
                body_angles = df_ztimed.loc[:,['propBoutIEI_pitch']].rename(columns={'propBoutIEI_pitch':f'exp{expNum}'}).transpose()
                all_angles = pd.concat([all_angles, body_angles])
                exp_date_match = pd.concat([exp_date_match, pd.DataFrame( data= {'expNum':expNum,'date':[exp[0:6]]} )],ignore_index=True)             

            # --- legacy method ---
            # jackknife for the index
            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            # get the distribution of every jackknifed sample for the current condition
            jack_y = pd.concat([pd.DataFrame(
                np.histogram(all_angles.iloc[idx_group].to_numpy().flatten(), bins=bins, density=True)
            ) for idx_group in jackknife_idx], axis=1).transpose()
                
            # combine conditions
            jack_y_all = pd.concat([jack_y_all, jack_y.assign(age=all_conditions[condition_idx][0:2], 
                                                            condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)
            # get the std of every jackknifed sample
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                ang_std.append(np.nanstd(all_angles.iloc[idx_group].to_numpy().flatten())) 
            ang_std = pd.DataFrame(ang_std).assign(excluded_exp=exp_date_match['date'])
            ang_std_all = pd.concat([ang_std_all, ang_std.assign(age=all_conditions[condition_idx][0:2], condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)
            # --- legacy method ---


jack_y_all.columns = ['Probability','Posture (deg)','dpf','condition']
jack_y_all.sort_values(by=['condition'],inplace=True)
ang_std_all.columns = ['std(posture)','excluded_exp','dpf','condition']                
ang_std_all.sort_values(by=['condition'],inplace=True)

# %%
# Stats
# # For multiple comparison
# multi_comp = MultiComparison(ang_std_all['std(posture)'], ang_std_all['dpf']+ang_std_all['condition'])
# print(multi_comp.tukeyhsd().summary())

# %%
# Plot posture distribution and its standard deviation

defaultPlotting()

# Separate data by age.
age_condition = set(jack_y_all['dpf'].values)
age_cond_num = len(age_condition)

# initialize a multi-plot, feel free to change the plot size
f, axes = plt.subplots(nrows=2, ncols=age_cond_num, figsize=(5*(age_cond_num), 10), sharey='row')
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
        # multiple comparison for more than 2 conditions
        print(f'* Age {age}:' )
        multi_comp = MultiComparison(ang_std_all['std(posture)'], ang_std_all['dpf']+ang_std_all['condition'])
        print(multi_comp.tukeyhsd().summary())
    else:
        pass
    
# plt.show()
plt.savefig(fig_dir+f"/{pick_data}'s IEIpitchMean_jackknife.pdf",format='PDF')


# %% plot percentage changes
for string in all_conditions:  # if has lesion data or not
    if string.find("esion") != -1:
        if_plt_percentage = 1
        break
    else:
        if_plt_percentage = 0
        
if if_plt_percentage == 1: # if contain lesion data    
    all_conditions.sort()
    ori_data = ang_std_all.loc[ang_std_all['dpf']==age]
    tmp = ori_data.groupby('condition').mean()
    lesion_mean = max(tmp.iloc[:,0])
    cond0_plt =  ori_data.loc[ori_data['condition'].str.find('ib') !=-1,:]
    cond1_plt =  ori_data.loc[ori_data['condition'].str.find('au') !=-1,:]

    cond0_plt = cond0_plt.sort_values(by=['excluded_exp']).reset_index()
    cond1_plt = cond1_plt.sort_values(by=['excluded_exp']).reset_index()

    pltData_cond0 = cond0_plt.loc[:,'std(posture)']
    pltData_cond1 = cond1_plt.loc[:,'std(posture)']
    percentage_chg = (pltData_cond0-pltData_cond1).divide(pltData_cond0-lesion_mean) *100

    percentage_chg = 100-percentage_chg
    cond0_plt = cond0_plt.assign(y = [100] * len(percentage_chg))
    cond1_plt = cond1_plt.assign(y = percentage_chg)
    plt_data = pd.concat([cond0_plt,cond1_plt])
    
    defaultPlotting()

    # Separate data by age.
    age_condition = set(jack_y_all['dpf'].values)
    age_cond_num = len(age_condition)

    # initialize a multi-plot, feel free to change the plot size
    f, axes = plt.subplots(nrows=2, ncols=age_cond_num, figsize=(5*(age_cond_num), 10), sharey='row')
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

        # plot std
        std_plt = plt_data.loc[plt_data['dpf']==age]
        # plot jackknifed paired data
        p = sns.pointplot(x='condition', y='y', hue='excluded_exp',data=std_plt,
                        palette=sns.color_palette(flatui), scale=0.5,
                        ax=axes[i+age_cond_num],
                    #   order=['Sibs','Tau','Lesion'],
        )
        # plot mean data
        p = sns.pointplot(x='condition', y='y',hue='condition',data=std_plt, 
                        linewidth=0,
                        alpha=0.9,
                        ci=None,
                        markers='d',
                        ax=axes[i+age_cond_num],
                        #   order=['Sibs','Tau','Lesion'],
        )
        p.legend_.remove()
        # p.set_yticks(np.arange(0.1,0.52,0.04))
        p.set_ylim(0,100)
        sns.despine(trim=False)
        
        plt.show()
        f.savefig(fig_dir+"/perct_chg.pdf",format='PDF')

# %%
