'''
This script plots bout frequency as a function of pitch angle
and fiitted coefs of function y = a * ((x-b)**2) + c (jackknife resampling is applied if contains data from multiple experiments)

This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, bout frequency as a function of pitch angle of the current experiment will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. if input directory is a folder with subfolders containing dlm data, jackknife resampling will be applied to calculate errors for coefs of fitted porabola. 
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)

# %%
def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean()
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit):
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
def plot_bout_frequency(root):
    print('\n- Plotting bout frequency as a function of pitch')
    # CONSTANTS
    BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
    X_RANGE_FULL = range(-60,61,1)
        
    folder_name = 'bout frequency'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced:')
        print(fig_dir)

    # %%
    # main function

    # Check if there are subfolders in the current directory
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        if_jackknife = True
        all_dir = all_dir[1:]
    else:
        # if no, only calculate one std(pitch) for current experiment
        if_jackknife = False

    # initialize results dataframe
    all_day_angles = pd.DataFrame()  # all ori pitch including all conditions, for validation 
    jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
    jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data


    # go through each condition folders under the root
    for expNum, exp_path in enumerate(all_dir):
        df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')               
        body_angles = df.loc[:,['propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime']]
        day_angles = day_night_split(body_angles,'propBoutIEItime').assign(exp_num=expNum)
        day_angles.dropna(inplace=True)
        all_day_angles = pd.concat([all_day_angles, day_angles[['propBoutIEI', 'propBoutIEI_pitch','exp_num']]],ignore_index=True)

    all_day_angles = all_day_angles.assign(y_boutFreq=1/all_day_angles['propBoutIEI'])

    binned_angles = distribution_binned_average(all_day_angles, BIN_WIDTH)

    # %%
    if if_jackknife:
        # jackknife for the index
        jackknife_idx = jackknife_resampling(np.arange(0,expNum+1))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_coef, this_fitted_y = parabola_fit1(all_day_angles.loc[all_day_angles['exp_num'].isin(idx_group)], X_RANGE_FULL)
            jackknifed_coef = pd.concat([jackknifed_coef, this_coef.assign(
                excluded_exp=excluded_exp
                )])
            jackknifed_y = pd.concat([jackknifed_y, this_fitted_y.assign(
                excluded_exp=excluded_exp
                )])
        jackknifed_coef.columns = ['sensitivity','x_inter','y_inter','jackknife_excluded_sample']
        jackknifed_coef['sensitivity'] = jackknifed_coef['sensitivity']*1000  # unit: mHz/deg**2
        jackknifed_y.columns = ['y','x','jackknife_excluded_sample']

    # fit angles condition by condition and concatenate results
    all_coef, all_fitted_y = parabola_fit1(all_day_angles, X_RANGE_FULL)
    all_coef.columns = ['sensitivity','x_inter','y_inter']
    all_fitted_y.columns = ['y','x']
        
    # %%
    print("Fitted coefs using ALL data (for reference):")
    print(all_coef)

    # %%
    # plot fitted parabola and sensitivity
    defaultPlotting()
    set_font_type()

    # loop through differrent age (dpf), plot parabola in the first row and sensitivy in the second.
    if if_jackknife:
        fitted = jackknifed_y
        coef_plt = jackknifed_coef
    else:
        fitted = all_fitted_y
        coef_plt = all_coef

    fitted.reset_index(drop=True,inplace=True)
    # dots are plotted with binned average pitches
    g = sns.lineplot(x='x',y='y',data=fitted, ci="sd")
    g = sns.scatterplot(x='propBoutIEI_pitch',y='y_boutFreq',s=20, data=binned_angles, alpha=0.3,linewidth=0)
    g.set_ylim(0, None,30)
    g.set_xlabel("Pitch angle")
    g.set_ylabel("Bout frequency (mHz/deg^2)")

    plt.savefig(os.path.join(fig_dir, "bout frequency vs posture.pdf"),format='PDF')

    # %%
    f, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))

    # SENSITIVITY
    g = sns.pointplot(y='sensitivity',data=coef_plt, 
                    linewidth=0,
                    alpha=0.9,
                    markers='d',
                    ax=axes[0],
    )
    # X INTERSECT
    sns.swarmplot(y='x_inter', data=coef_plt,
                ax=axes[1],
                )
    # Y INTERSECT
    sns.swarmplot(y='y_inter', data=coef_plt,
                ax=axes[2])
    f.savefig(os.path.join(fig_dir, "sensitivity and other coef.pdf"),format='PDF')

if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_bout_frequency(root)