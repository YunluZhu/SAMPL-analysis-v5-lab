'''
This script plots bout frequency as a function of IBI pitch angle
and fiitted coefs of function y = a * ((x-b)**2) + c  

This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, bout frequency as a function of pitch angle of the current experiment will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
    
2. if input directory is a folder with subfolders containing dlm data
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5

NOTE
User may define the number of bouts sampled from each experimental repeat by defining the argument "sample_bout"
Default is off (sample_bout = -1)
'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from plot_functions.plt_tools import (set_font_type, day_night_split, round_half_up, setup_vis_parameter, defaultPlotting)

# %%
def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-60,60,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','boutFreq']].mean()
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['boutFreq'], p0=(0.0002,3,0.8) , bounds=((0, -10, 0),(0.5, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %%
def plot_bout_timing(root:str, **kwargs):
    """plots bout frequency as a function of IBI pitch angle

    Args:
        root (stirng): directory containing analyzed .dlm or folders containing analyzed .dlm representing experimental repeats
        ---kwargs---
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off
        figure_dir (str): directory to save figures. If not defined, figures will be saved to folder "figures"
    """
    print('------\n+ Plotting bout frequency as a function of pitch')
    # CONSTANTS
    BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
    X_RANGE_FULL = range(-60,61,1)
    
    folder_name = 'bout timing'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    root, all_dir, fig_dir, if_sample, SAMPLE_N, if_multiple_rep = setup_vis_parameter(root, fig_dir, if_sample=False, SAMPLE_N=-1, if_multiple_repeats=False, **kwargs)

    # %%
    # main function

    # initialize results dataframe
    all_day_angles = pd.DataFrame()  # all ori pitch including all conditions, for validation 
    coef_rep = pd.DataFrame()  
    y_rep = pd.DataFrame()  

    # go through each condition folders under the root
    for expNum, exp_path in enumerate(all_dir):
        df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')               
        body_angles = df.loc[:,['propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime']]
        day_angles = day_night_split(body_angles,'propBoutIEItime').assign(expNum=expNum)
        day_angles.dropna(inplace=True)
        
        if if_multiple_rep == True:
            if if_sample == True:
                try:
                    day_angles = day_angles.sample(n=SAMPLE_N)
                except:
                    day_angles = day_angles.sample(n=SAMPLE_N,replace=True)

        all_day_angles = pd.concat([all_day_angles, day_angles[['propBoutIEI', 'propBoutIEI_pitch','expNum']]],ignore_index=True)

    all_day_angles = all_day_angles.assign(boutFreq=1/all_day_angles['propBoutIEI'])
    binned_angles = distribution_binned_average(all_day_angles, BIN_WIDTH)
    # print(f"Mean bout frequency: {all_day_angles.mean().loc['boutFreq']:.3f}±{all_day_angles.std().loc['boutFreq']:.3f}")
    # print(f"Mean bout interval: {all_day_angles.mean().loc['propBoutIEI']:.3f}±{all_day_angles.std().loc['propBoutIEI']:.3f}") 
    # print(f"Mean bout pitch: {all_day_angles.mean().loc['propBoutIEI_pitch']:.3f}±{all_day_angles.std().loc['propBoutIEI_pitch']:.3f}")

    data_return = all_day_angles.loc[:,['propBoutIEI', 'propBoutIEI_pitch','boutFreq']].describe()
    
    # %%
    if if_multiple_rep:
        rep_idx = all_day_angles['expNum'].unique()
        for repeat in rep_idx:
            this_coef, this_fitted_y = parabola_fit1(all_day_angles.loc[all_day_angles['expNum']==repeat], X_RANGE_FULL)
            coef_rep = pd.concat([coef_rep, this_coef.assign(
                expNum=repeat
                )])
            y_rep = pd.concat([y_rep, this_fitted_y.assign(
                expNum=repeat
                )])
        coef_rep.columns = ['sensitivity','x_inter','y_inter','expNum'] 
        coef_rep['sensitivity'] = coef_rep['sensitivity']*1000  # unit: mHz/deg**2
        y_rep.columns = ['y','x','expNum']

    # fit angles condition by condition and concatenate results
    all_coef, all_fitted_y = parabola_fit1(all_day_angles, X_RANGE_FULL)
    all_coef.columns = ['sensitivity','x_inter','y_inter']
    all_fitted_y.columns = ['y','x']
    all_coef['sensitivity'] = all_coef['sensitivity']*1000
    

    # %%
    # plot fitted parabola and sensitivity
    
    boutData_filename = os.path.join(fig_dir,f"IBI values.csv")

    if if_multiple_rep:
        coef_filename = os.path.join(fig_dir,f"bout timing coef mean values (repeats).csv")
        output_coef = coef_rep.iloc[:,[0,1,2]].describe()
    else:
        output_coef = all_coef
        coef_filename = os.path.join(fig_dir,f"bout timing coef mean values.csv")
        
    print("Fitted coef:")
    print(output_coef)
    output_coef.to_csv(coef_filename)
    print(data_return)
    data_return.to_csv(boutData_filename)


    defaultPlotting()
    set_font_type()
    # loop through differrent age (condition0), plot parabola in the first row and sensitivy in the second.
    if if_multiple_rep:
        fitted = y_rep
        coef_plt = coef_rep
    else:
        fitted = all_fitted_y
        coef_plt = all_coef

    
    fitted.reset_index(drop=True,inplace=True)
    # dots are plotted with binned average pitches
    g = sns.lineplot(x='x',y='y',data=fitted, errorbar="sd")
    g = sns.scatterplot(x='propBoutIEI_pitch',y='boutFreq',s=20, data=binned_angles, alpha=0.3,linewidth=0)
    g.set_ylim(0, None,30)
    g.set_xlabel("Pitch angle")
    g.set_ylabel("Bout frequency (mHz/deg^2)")

    plt.savefig(os.path.join(fig_dir, "bout frequency vs posture.pdf"),format='PDF')

    # %%
    f, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))

    # SENSITIVITY
    g = sns.pointplot(y='sensitivity',data=coef_plt, 
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
    plot_bout_timing(root)