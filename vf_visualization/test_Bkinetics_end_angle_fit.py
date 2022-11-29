'''
new function, still working on what we can learn from this

plot post angvel vs. end pitch, fit with a porabola

zeitgeber time? Yes
'''

#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'sfld_combined'
which_ztime = 'all'
RESAMPLE = 0



root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'B_end_ang_fit_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
# CONSTANTS
# SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
X_RANGE_FULL = range(-60,61,1)

def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='pitch_end')
    # df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['pitch_end'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['pitch_end','angvel_post_phase']].mean()
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit = X_RANGE_FULL):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['pitch_end'], df['angvel_post_phase'], 
                           p0=(-0.005,3,3) , 
                        #    bounds=((0, -5, 0),(10, 15, 10))
                           )
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

# get kinetics for separating up and down
all_kinetics = all_feature_cond.groupby(['dpf']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
all_feature_UD = pd.DataFrame()
all_feature_cond = all_feature_cond.assign(direction=np.nan)
for key, group in all_feature_cond.groupby(['dpf']):
    this_setvalue = all_kinetics.loc[all_kinetics['dpf']==key,'set_point'].to_list()[0]
    print(this_setvalue)
    group['direction'] = pd.cut(group['pitch_initial'],
                                bins=[-91,this_setvalue,91],
                                labels=['dn','up'])
    all_feature_UD = pd.concat([all_feature_UD,group])
    
# %% jackknife and fit
jackknifed_coef = pd.DataFrame()
jackknifed_y = pd.DataFrame()
binned_angles = pd.DataFrame()

cat_cols = ['condition','dpf','ztime']

if RESAMPLE == 0:
    sampled_angles = all_feature_UD
else:
    sampled_angles = all_feature_UD.groupby(['condition','dpf','ztime','exp']).sample(
        n=RESAMPLE,
        replace=True,
        )
for (this_cond, this_dpf, this_ztime), group in sampled_angles.groupby(cat_cols):
    jackknife_idx = jackknife_resampling(np.array(list(range(sampled_angles['expNum'].max()+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_df_toFit = group.loc[group['expNum'].isin(idx_group),['angvel_post_phase','pitch_end']].reset_index(drop=True)
        this_df_toFit.dropna(inplace=True)
        coef, fitted_y = parabola_fit1(this_df_toFit, X_RANGE_FULL)
        jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=this_dpf,
                                                                condition=this_cond,
                                                                excluded_exp=excluded_exp,
                                                                ztime=this_ztime)])
        jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=this_dpf,
                                                                condition=this_cond,
                                                                excluded_exp=excluded_exp,
                                                                ztime=this_ztime)])
        
    this_binned_angles = distribution_binned_average(this_df_toFit, BIN_WIDTH)
    this_binned_angles = this_binned_angles.assign(dpf=this_dpf,
                                                    condition=this_cond,
                                                    ztime=this_ztime)
    binned_angles = pd.concat([binned_angles, this_binned_angles],ignore_index=True)

# %%
jackknifed_y.columns = ['post angvel','end pitch','dpf','condition','jackknife num','ztime']
jackknifed_y = jackknifed_y.reset_index(drop=True)

jackknifed_coef.columns = ['a','x intersect','y intersect','dpf','condition','jackknife num','ztime']
jackknifed_coef = jackknifed_coef.reset_index(drop=True)

binned_angles = binned_angles.reset_index(drop=True)

all_ztime = list(set(jackknifed_coef['ztime']))
all_ztime.sort()
# %% plot
g = sns.relplot(x='end pitch',y='post angvel', data=jackknifed_y, 
                kind='line',
                col='dpf', col_order=all_cond1,
                row = 'ztime', row_order=all_ztime,
                hue='condition', hue_order = all_cond2,ci='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=binned_angles.loc[
            (binned_angles['dpf']==all_cond1[j]) & (binned_angles['ztime']==all_ztime[i]),:
                ], 
                    x='pitch_end', y='angvel_post_phase', 
                    hue='condition',alpha=0.2,
                    ax=ax)
g.set(xlim=(-30, 40),
      ylim=(-12,10))
    
filename = os.path.join(fig_dir,f"end angvel fit sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')



# %%
# plot all coef

plt.close()
col_to_plt = {0:'a',1:'x intersect',2:'y intersect'}
for i in np.arange(len(coef.columns)):
    p = sns.catplot(
        data = jackknifed_coef, y=col_to_plt[i],x='dpf',kind='point',join=False,
        col_order=all_cond1,ci='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='condition', dodge=True,
        hue_order = all_cond2,sharey=False
    
    )
    p.map(sns.lineplot,'dpf',col_to_plt[i],estimator=None,
        units='jackknife num',
        hue='condition',
        alpha=0.2,
        data=jackknifed_coef)
    filename = os.path.join(fig_dir,f"end angvel coef{i} sample{RESAMPLE}.pdf")
    plt.savefig(filename,format='PDF')
# %%
