#%%
# import sys
# from ensurepip import bootstrap
import os
from plot_functions.plt_tools import round_half_up
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
from sklearn.metrics import r2_score

# %%
def sampling_regression_rep(df, list_of_sample_N, regression_func, bootstrap):
    num_of_sampling_repeats = 20
    sampling_rep = 0
    value_list = []
    rep_list = []
    fit_rsq = []
    sigma_list = []
    sample_N_list = []
    sampling_rep_list = []
    
    if bootstrap:
        num_of_repeats = 20
    else:
        num_of_repeats = 1
        
    while sampling_rep < num_of_sampling_repeats:
        for sample_N in list_of_sample_N:
            sampled_df = df.sample(n=sample_N)
            rep = 0
            while rep < num_of_repeats: # bootstrap sampling to estimate fit robustness
                if bootstrap:
                    sample_for_fit = sampled_df.iloc[np.random.randint(sample_N, size=sample_N)]
                else:
                    sample_for_fit = sampled_df
                sample_for_fit.reset_index(drop=True, inplace=True)
                value, r_squared, sigma = regression_func(sample_for_fit, sample_N)
                fit_rsq.append(r_squared)
                value_list.append(value)
                sigma_list.append(sigma)
                rep_list.append(rep)
                sampling_rep_list.append(sampling_rep)
                sample_N_list.append(sample_N)
                rep+=1
        sampling_rep+=1

    df_tocalc = pd.DataFrame(data={
        'values': value_list,
        'rep': rep_list,
        'nobs': sample_N_list,
        'sigma': sigma_list,
        'r_square': fit_rsq,
        'sampling_rep': sampling_rep_list,
    })
    return df_tocalc

def plot_model_R2_confidence_interval(df_tocalc, deg_free, nobs_col = 'nobs', r_square_col = 'r_square', fig_name = 'R2 CI width vs Bout Number'):
    alpha_list = [0.05, 0.01, 0.001]
    res = pd.DataFrame()
    for sampling_rep, df_sample in df_tocalc.groupby('sampling_rep'):
        for alpha in alpha_list:
            R2 = []
            cohens_f2 = []
            nobs_list = []
            # power = []
            ci_width = []
            ci_half_width_percent = []
            for nobs, group in df_sample.groupby(nobs_col):
                # mean_val = group['values'].mean()
                meanR2 = group[r_square_col].mean()
                f2_rep = group[r_square_col]/(1 - group[r_square_col])
                es = f2_rep.mean()
                cohens_f2.append(f2_rep.mean())
                std_err = st.sem(group[r_square_col])
                # solver = FTestPower()
                # solved_res = solver.solve_power(effect_size=np.sqrt(es), df_denom=deg_free - 1, df_num = nobs - deg_free, nobs=nobs, alpha=alpha, power=None, ncc=1) # Updated function
                # power.append(solved_res)
                R2.append(meanR2)
                nobs_list.append(nobs)
                (ci_low, ci_high) = st.t.interval(1-alpha, 20-1, loc=meanR2, scale=std_err)
                ci_width.append(ci_high-ci_low)
                ci_half_width_percent.append((ci_high-ci_low)*100/2/meanR2)

            to_concat = pd.DataFrame (data={
                # 'power': power,
                'cohensF2': cohens_f2,
                'R2': R2,
                'bout number': nobs_list,
                'alpha': alpha,
                'ci_width': ci_width,
                'ci_half_width_percent': ci_half_width_percent,
                'sampling_rep': sampling_rep,
                })
            res = pd.concat([res,to_concat],ignore_index=True)
    plt.figure(figsize=(4,3))
    g = sns.lineplot(
        data = res,
        x = 'bout number',
        y = 'ci_width',
        hue = 'alpha',
        legend='full',
        errorbar = None
    )
    sns.despine()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    g.set(title=fig_name)
    ci_fig = get_figure_dir('Fig_8')
    filename = os.path.join(ci_fig,fig_name+".pdf")
    plt.savefig(filename,format='PDF')
    return res
   
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit2(df, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    ydata = df['bout_freq']
    xdata = df['propBoutIEI_pitch']
    popt, pcov = curve_fit(ffunc1, xdata, ydata, 
                           p0=(0.005,3,0.5) , 
                           bounds=((0, -5, 0),(10, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(cond1=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    r_squared = r2_score(ydata, ffunc1(xdata, *popt))
    return output_coef, output_fitted, p_sigma, r_squared

def sigmoid_fit2(df, x_range_to_fit,func,**kwargs):
    ydata = df['atk_ang']
    xdata = df['rot_to_max_angvel']
    lower_bounds = [0.1,0,-100,1]
    upper_bounds = [10,20,2,100]
    x0=[5, 1, 0, 5]
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, xdata, ydata, 
                        #    maxfev=2000, 
                           p0 = p0,
                           bounds=(lower_bounds,upper_bounds))
    y = func(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    # Efron’s psudo R-squared
    y_pred = func(xdata,*popt)
    n = float(len(y))
    t1 = np.sum(np.power(ydata - y_pred, 2.0))
    t2 = np.sum(np.power((ydata - (np.sum(y) / n)), 2.0))
    EfronsR2 = 1.0 - (t1 / t2)
    return output_coef, output_fitted, p_sigma, EfronsR2

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

def IBI_timging_regression_stats(df, sample_N):
    X_RANGE_FULL = range(-30,41,1)
    coef, fitted_y, sigma, r_squared = parabola_fit2(df, X_RANGE_FULL)
    E_sensitivity = coef.iloc[0,0] * 1000
    sigma_sensitivity = sigma[0] * 1000
    p = 3
    R2 = r_squared
    r_squared_adj = 1 - (1 - R2) * (sample_N - 1) / (sample_N - p - 1)
    return E_sensitivity, r_squared_adj, sigma_sensitivity

def steering_regression_stats(df, sample_N):
    xcol = 'traj_peak'
    ycol = 'pitch_peak'
    xdata = df[xcol] 
    ydata = df[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    p = 2
    R2 = r_value**2
    r_squared_adj = 1 - (1 - R2) * (sample_N - 1) / (sample_N - p - 1)
    return slope, r_squared_adj, std_err 

def righting_regression_stats(df, sample_N):
    xcol = 'pitch_initial'
    ycol = 'rot_l_decel'
    xdata = df[xcol] 
    ydata = df[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    p = 2
    R2 = r_value**2
    r_squared_adj = 1 - (1 - R2) * (sample_N - 1) / (sample_N - p - 1)
    return slope*(-1), r_squared_adj, std_err 

def fin_body_regression_stats(df, sample_N):
    X_RANGE = np.arange(-2,8.01,0.01)
    coef, fitted_y, sigma, psudo_r_squared = sigmoid_fit2(
        df, X_RANGE, func=sigfunc_4free
    )
    E_height = coef.iloc[0,3]
    E_k = coef.iloc[0,0] 
    V_height = sigma[3]**2  # height error
    V_k = sigma[0]**2  # k error
    slope_var = (V_height*V_k + V_height*(E_k**2) +  V_k*(E_height**2)) * (1/4)**2  # estimate slope variance 
    slope = E_k * E_height / 4
    return slope, psudo_r_squared, np.sqrt(slope_var)

# %%
def Fig8_modeling_stats(root):
    """ Estimate regression robustness at given number of bouts

    Steps:
    1. determine list of N for bouts to sample
    2. sampling_regression_rep() samples bouts and calculates regression stats
        2.1 repeat 20 times for each N in the list of bouts to sample
        2.2 sample N bouts from the dataset
        2.3 bootstrap the same sampled dataset, run following steps for 20 times, if not, use the raw sampled dataset, run following steps once
        2.4 call regression_func(), calculate value, R2, sigma for value
    3. run function plot_model_R2_confidence_interval(), which:
        3.0. for each alpha given
        3.1. Calculates mean R2 from 20x repeats for each bout number
        3.2. Calculates sem of R2
        3.3. Calculates CI width for R2
        3.4. plot CI width as a function of bout number
    """
    
    set_font_type()
    mpl.rc('figure', max_open_warning = 0)
    
    # Select data and create figure folder
    which_ztime = 'day'
    FRAME_RATE = 166
    
    ci_fig = get_figure_dir('Fig_8')

    try:
        os.makedirs(ci_fig)
    except:
        pass

    all_feature_cond, all_cond0, all_cond0 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
    all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
    IBI_angles, cond1_all, cond1_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

    print("- Figure 8: bout timing modeling & sensitivity")
    IBI_angles.dropna(inplace=True)

    list_of_sample_N = np.linspace(500,len(IBI_angles)//1000 * 1000,8).astype(int)

    df_tocalc = sampling_regression_rep(IBI_angles, list_of_sample_N, IBI_timging_regression_stats, bootstrap=True)
    power = plot_model_R2_confidence_interval(df_tocalc, 3, fig_name='Bout timing parabola modeling - R2 CI by alpha')

    print("- Figure 8: Steering fit & steering gain")

    list_of_sample_N = np.linspace(500,len(all_feature_cond)//1000*1000,8).astype(int)
    df_tocalc = sampling_regression_rep(all_feature_cond, list_of_sample_N, steering_regression_stats, bootstrap=True)
    power = plot_model_R2_confidence_interval(df_tocalc, 2, fig_name='Steering fit - R2 CI by alpha')

    print("- Figure 8: Righting fit & righting gain")

    list_of_sample_N = np.linspace(500,len(all_feature_cond)//1000*1000,8).astype(int)
    df_tocalc = sampling_regression_rep(all_feature_cond, list_of_sample_N, righting_regression_stats, bootstrap=True)
    power = plot_model_R2_confidence_interval(df_tocalc, 2, fig_name='Righting fit - R2 CI by alpha')

    print("- Figure 8: Fin-body coordination")

    # for righting gain
    bouts_to_plot = all_feature_cond.loc[all_feature_cond['spd_peak']>=7]

    list_of_sample_N = np.linspace(2000,len(bouts_to_plot)//1000*1000,8).astype(int)
    df_tocalc = sampling_regression_rep(bouts_to_plot, list_of_sample_N, fin_body_regression_stats, bootstrap=True)
    power = plot_model_R2_confidence_interval(df_tocalc, 4, fig_name='Fin-body modeling - R2 CI by alpha')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_combined'? \n")
    Fig8_modeling_stats(root)