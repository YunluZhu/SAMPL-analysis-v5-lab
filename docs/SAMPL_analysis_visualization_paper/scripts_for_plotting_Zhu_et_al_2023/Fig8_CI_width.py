#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type)
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.stats import linregress
import scipy.stats as st

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%

def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['bout_freq'], 
                           p0=(0.005,3,0.5) , 
                           bounds=((0, -5, 0),(10, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))

    return output_coef, output_fitted, p_sigma

def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,0,-100,1]
    upper_bounds = [10,20,2,100]
    x0=[5, 1, 0, 5]
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df['rot_to_max_angvel'], df['atk_ang'], 
                        #    maxfev=2000, 
                           p0 = p0,
                           bounds=(lower_bounds,upper_bounds))
    y = func(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

# %%
def Fig8_CI_width(root):
    # Select data and create figure folder
    which_ztime = 'day'

    FRAME_RATE = 166
    number_of_N = 40

    ci_fig = get_figure_dir('Fig_8')

    try:
        os.makedirs(ci_fig)
    except:
        pass

    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])


    print("- Figure 8: CI width vs sample size - bout timing sensitivity")
    X_RANGE_FULL = range(-30,41,1)
    list_of_sample_N = np.linspace(1000,len(IBI_angles),number_of_N).astype(int)
    repeated_res = pd.DataFrame()
    num_of_repeats = 20
    rep = 0

    while rep < num_of_repeats:
        list_of_ci_width = []
        for sample_N in list_of_sample_N:
            sample_for_fit = IBI_angles.sample(n=sample_N, replace=True)[['propBoutIEI_pitch','bout_freq','propBoutIEI']]
            sample_for_fit.dropna(inplace=True)
            coef, fitted_y, sigma = parabola_fit1(sample_for_fit, X_RANGE_FULL)
            E_sensitivity = coef.iloc[0,0] * 1000
            sigma_sensitivity = sigma[0] * 1000
                    
            (ci_low, ci_high) = st.norm.interval(0.95, loc=E_sensitivity, scale=sigma_sensitivity)
            ci_width = ci_high - ci_low
            list_of_ci_width.append(ci_width)
        res = pd.DataFrame(
            data = {
                'sample':list_of_sample_N,
                'CI width': list_of_ci_width,
            }
        )
        repeated_res = pd.concat([repeated_res,res],ignore_index=True)
        rep+=1

    plt.figure(figsize=(5,4))
    g = sns.lineplot(
        data = repeated_res,
        x = 'sample',
        y = 'CI width',
        errorbar='sd',
        color = 'olive'
    )
    sns.despine()
    g.set_xlim(0,max(list_of_sample_N))
    g.set(title='Sensitivity')
    filename = os.path.join(ci_fig,"bout timing sensitivity CI width.pdf")
    plt.savefig(filename,format='PDF')



    print("- Figure 8: CI width vs sample size - Steering gain")

    # for steering gain
    list_of_sample_N = np.linspace(1000,len(all_feature_cond),number_of_N).astype(int)
    repeated_res = pd.DataFrame()
    num_of_repeats = 20
    rep = 0

    xcol = 'traj_peak'
    ycol = 'pitch_peak'

    while rep < num_of_repeats:
        list_of_ci_width = []
        for sample_N in list_of_sample_N:
            sample_for_fit = all_feature_cond.sample(n=sample_N, replace=True)
            sample_for_fit.dropna(inplace=True)
            xdata = sample_for_fit[xcol] 
            ydata = sample_for_fit[ycol]
            model_par = linregress(xdata, ydata)
            slope, intercept, r_value, p_value, std_err = model_par
            (ci_low, ci_high) = st.norm.interval(0.95, loc=slope, scale=std_err)
            ci_width = ci_high - ci_low
            list_of_ci_width.append(ci_width)
        res = pd.DataFrame(
            data = {
                'sample':list_of_sample_N,
                'CI width': list_of_ci_width,
            }
        )
        repeated_res = pd.concat([repeated_res,res],ignore_index=True)
        rep+=1

    plt.figure(figsize=(5,4))
    g = sns.lineplot(
        data = repeated_res,
        x = 'sample',
        y = 'CI width',
        errorbar = 'sd',
        color = 'darkgreen'
    )
    sns.despine()
    g.set_xlim(0,max(list_of_sample_N))
    g.set(title='Steering gain')
    filename = os.path.join(ci_fig,"Steering gain CI width.pdf")
    plt.savefig(filename,format='PDF')


    # plot CI of slope
    print("- Figure 8: CI width vs sample size - max slope of fin-body ratio")
    X_RANGE = np.arange(-2,8.01,0.01)
    bouts_to_plot = all_feature_cond.loc[all_feature_cond['spd_peak']>=7]
    list_of_sample_N = np.linspace(4000,len(bouts_to_plot),number_of_N).astype(int)

    repeated_res = pd.DataFrame()
    num_of_repeats = 20
    rep = 0

    while rep < num_of_repeats:
        list_of_ci_width = []
        for sample_N in list_of_sample_N:
            sample_for_fit = bouts_to_plot.sample(n=sample_N, replace=True)
            sample_for_fit.drop(sample_for_fit.loc[(sample_for_fit['atk_ang']<0) & (sample_for_fit['rot_steering']>sample_for_fit['rot_steering'].median())].index)
            coef, fitted_y, sigma = sigmoid_fit(
                sample_for_fit, X_RANGE, func=sigfunc_4free
            )
            E_height = coef.iloc[0,3]
            E_k = coef.iloc[0,0] 
            V_height = sigma[3]**2
            V_k = sigma[0]**2
            
            mean_formSample = E_k * E_height / 4
            
            slope_var = (V_height*V_k + V_height*(E_k**2) +  V_k*(E_height**2)) * (1/4)**2
            sigma_formSample = np.sqrt(slope_var)
            
            (ci_low, ci_high) = st.norm.interval(0.95, loc=mean_formSample, scale=sigma_formSample)
            ci_width = ci_high - ci_low
            list_of_ci_width.append(ci_width)
        res = pd.DataFrame(
            data = {
                'sample':list_of_sample_N,
                'CI width': list_of_ci_width,
            }
        )
        repeated_res = pd.concat([repeated_res,res],ignore_index=True)
        rep+=1

    plt.figure(figsize=(5,4))
    g = sns.lineplot(
        data = repeated_res,
        x = 'sample',
        y = 'CI width',
        errorbar='sd',
        color='purple'
    )
    sns.despine()
    g.set_xlim(0,max(list_of_sample_N))
    g.set(title='Fin-body ratio')
    filename = os.path.join(ci_fig,"fin-body ratio slope CI width.pdf")
    plt.savefig(filename,format='PDF')

    print("- Figure 8: CI width vs sample size - Righting gain")

    # for righting gain
    list_of_sample_N = np.linspace(1000,len(all_feature_cond),number_of_N).astype(int)
    repeated_res = pd.DataFrame()
    num_of_repeats = 20
    rep = 0

    xcol = 'pitch_initial'
    ycol = 'rot_righting'

    while rep < num_of_repeats:
        list_of_ci_width = []
        for sample_N in list_of_sample_N:
            sample_for_fit = all_feature_cond.sample(n=sample_N, replace=True)
            sample_for_fit.dropna(inplace=True)
            xdata = sample_for_fit[xcol] 
            ydata = sample_for_fit[ycol]
            model_par = linregress(xdata, ydata)
            slope, intercept, r_value, p_value, std_err = model_par
            (ci_low, ci_high) = st.norm.interval(0.95, loc=slope, scale=std_err)
            ci_width = ci_high - ci_low
            list_of_ci_width.append(ci_width)
        res = pd.DataFrame(
            data = {
                'sample':list_of_sample_N,
                'CI width': list_of_ci_width,
            }
        )
        repeated_res = pd.concat([repeated_res,res],ignore_index=True)
        rep+=1

    plt.figure(figsize=(5,4))
    g = sns.lineplot(
        data = repeated_res,
        x = 'sample',
        y = 'CI width',
        errorbar = 'sd',
        color = 'red'
    )
    sns.despine()
    g.set_xlim(0,max(list_of_sample_N))
    g.set(title='Righting gain')
    filename = os.path.join(ci_fig,"Righting gain CI width.pdf")
    plt.savefig(filename,format='PDF')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig8_CI_width(root)