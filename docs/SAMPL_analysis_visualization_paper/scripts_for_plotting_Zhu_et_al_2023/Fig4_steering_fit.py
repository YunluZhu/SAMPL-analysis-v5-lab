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
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
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

def Fig4_steering_fit(root):
    set_font_type()
    mpl.rc('figure', max_open_warning = 0)

    which_ztime = 'day'
    FRAME_RATE = 166

    folder_name = f'Steering fit'

    fig_dir4 = os.path.join(get_figure_dir('Fig_4'), folder_name)

    try:
        os.makedirs(fig_dir4)
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
    print("Figure 4: Linear regression for steering")
    # steering
    print("- Steering Gain")
    xcol = 'traj_peak'
    ycol = 'pitch_peak'
    xmin = -30
    xmax = 50
    color = 'darkgreen'
    g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

    print(f"Pearson's correlation coefficient = {r_value}")
    print(f"Slope = {slope}")
    print(f"Steering gain = {slope}")
    

    r_squared = r2_score(toplt['pitch_peak'], linear_mod(toplt['traj_peak'], slope.mean(), intercept.mean()))
    print(f"r-squared of linear fit: {r_squared:3f}")
    

    g.set_xlabel(xcol+' (deg)')
    g.set_ylabel(ycol+' (deg)')
    g.set(
        ylim=(-30,50),
        xlim=(xmin,xmax)
        )
    plt.savefig(fig_dir4+"/Steering fit.pdf",format='PDF')
    
    
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig4_steering_fit(root)