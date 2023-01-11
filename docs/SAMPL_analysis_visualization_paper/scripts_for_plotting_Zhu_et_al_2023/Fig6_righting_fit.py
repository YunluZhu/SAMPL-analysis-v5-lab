#%%
# import sys
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (set_font_type)
import matplotlib as mpl
from scipy.stats import linregress
from sklearn.metrics import r2_score


def linReg_sampleSatter_plot(data,xcol,ycol,xmin,xmax,color):
    xdata = data[xcol] 
    ydata = data[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    x = np.linspace(xmin,xmax,100)
    y = slope*x+intercept
    plt.figure(figsize=(4,4))
    g = sns.scatterplot(x=xcol, 
                        y=ycol, 
                        data=data.sample(2000), 
                        # marker='+',
                        alpha = 0.1,
                        color='grey',
                        edgecolor="none",
                        )
    plt.plot(x, y, color=color)
    return g, slope, intercept, r_value, p_value, std_err

def linear_mod(x,slope,intercept): 
    y = slope*x+intercept
    return y


def Fig6_righting_fit(root):
    set_font_type()
    mpl.rc('figure', max_open_warning = 0)

    # %%
    # Select data and create figure folder

    which_ztime = 'day'
    FRAME_RATE = 166
    folder_name = f'Righting fit'

    # fig_dir4 = os.path.join(get_figure_dir('Fig_4'), folder_name)
    fig_dir6 = os.path.join(get_figure_dir('Fig_6'), folder_name)

    try:
        os.makedirs(fig_dir6)
    except:
        pass

    # %% get features
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
    # %% tidy data

    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

    all_feature_UD = all_feature_cond

    # parameter distribution
    toplt = all_feature_UD

    # %%
    print("- Figure 6: Linear regression for righting")
    # righting
    print("- Righting Gain")
    xcol = 'pitch_initial'
    ycol = 'rot_righting'
    xmin = -30
    xmax = 50
    color = 'darkred'
    g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

    print(f"Pearson's correlation coefficient = {r_value}")
    print(f"Slope = {slope}")
    print(f"Righting gain = {-1*slope}")
    
    r_squared = r2_score(toplt[ycol], linear_mod(toplt[xcol], slope, intercept))
    print(f"r-squared of linear fit: {r_squared:3f}")

    g.set_xlabel(xcol+' (deg)')
    g.set_ylabel(ycol+' (deg)')
    g.set(
        ylim=(-6,10),
        xlim=(xmin,xmax)
        )
    plt.savefig(os.path.join(fig_dir6,"Righting fit.pdf"),format='PDF')
    

    # set point
    print("- Set Point")
    xcol = 'pitch_initial'
    ycol = 'rot_righting'
    xmin = -30
    xmax = 50
    color = 'black'
    g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

    x_intercept = -1*intercept/slope
    print(f"Pearson's correlation coefficient = {r_value}")
    print(f"Set point: {x_intercept}")
    
    g.set_xlabel(xcol+' (deg)')
    g.set_ylabel(ycol+' (deg)')
    g.set(
        ylim=(-6,10),
        xlim=(xmin,xmax)
        )
    plt.hlines(y=0,xmin=-30,xmax=x_intercept,colors='darkgrey')
    plt.vlines(x=x_intercept,ymin=-6,ymax=0,colors='blue')
    plt.savefig(os.path.join(fig_dir6,"Set point.pdf"),format='PDF')
    

    # %%
    # Righting: posture distribution
    toplt = all_feature_UD
    feature_to_plt = ['pitch_initial','pitch_post_bout']
    upper = np.percentile(toplt['pitch_initial'], 99.5)
    lower = np.percentile(toplt['pitch_initial'], 0.5)
    for feature in feature_to_plt:
        # let's add unit
        if 'spd' in feature:
            xlabel = feature + " (mm*s^-1)"
        elif 'dis' in feature:
            xlabel = feature + " (mm)"
        else:
            xlabel = feature + " (deg)"

        plt.figure(figsize=(3,2))
        g = sns.histplot(data=toplt, x=feature, 
                            bins = 30, 
                            element="poly",
                            #  kde=True, 
                            stat="probability",
                            pthresh=0.05,
                            binrange=(lower,upper),
                            color='grey'
                            )
        g.set_xlabel(xlabel)
        sns.despine()
        plt.savefig(os.path.join(fig_dir6,f"{feature} distribution.pdf"),format='PDF')
        # plt.close()

    # %% plot righting distribution
    toplt = all_feature_UD
    feature = 'rot_righting'
    upper = np.percentile(toplt[feature], 99.5)
    lower = np.percentile(toplt[feature], 0.5)

    if 'spd' in feature:
        xlabel = feature + " (mm*s^-1)"
    elif 'dis' in feature:
        xlabel = feature + " (mm)"
    else:
        xlabel = feature + " (deg)"

    plt.figure(figsize=(3,2))
    g = sns.histplot(data=toplt, x=feature, 
                        bins = 20, 
                        element="poly",
                        #  kde=True, 
                        stat="probability",
                        pthresh=0.05,
                        binrange=(lower,upper),
                        color='grey'
                        )
    g.set_xlabel(xlabel)
    sns.despine()
    plt.savefig(os.path.join(fig_dir6,f"{feature} distribution.pdf"),format='PDF')
    # plt.close()
    # %%
    # # for steering gain

    # list_of_sample_N = np.arange(1000,len(all_feature_UD),500)
    # repeated_res = pd.DataFrame()
    # num_of_repeats = 20
    # rep = 0

    # xcol = 'traj_peak'
    # ycol = 'pitch_peak'

    # while rep < num_of_repeats:
    #     list_of_ci_width = []
    #     for sample_N in list_of_sample_N:
    #         sample_for_fit = all_feature_UD.sample(n=sample_N)
    #         sample_for_fit.dropna(inplace=True)
    #         xdata = sample_for_fit[xcol] 
    #         ydata = sample_for_fit[ycol]
    #         model_par = linregress(xdata, ydata)
    #         slope, intercept, r_value, p_value, std_err = model_par
    #         (ci_low, ci_high) = st.norm.interval(0.95, loc=slope, scale=std_err)
    #         ci_width = ci_high - ci_low
    #         list_of_ci_width.append(ci_width)
    #     res = pd.DataFrame(
    #         data = {
    #             'sample':list_of_sample_N,
    #             'CI width': list_of_ci_width,
    #         }
    #     )
    #     repeated_res = pd.concat([repeated_res,res],ignore_index=True)
    #     rep+=1

    # plt.figure(figsize=(5,4))
    # g = sns.lineplot(
    #     data = repeated_res,
    #     x = 'sample',
    #     y = 'CI width'
    # )
    # filename = os.path.join(ci_fig,"Steering gain CI width.pdf")
    # plt.savefig(filename,format='PDF')
    
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig6_righting_fit(root)