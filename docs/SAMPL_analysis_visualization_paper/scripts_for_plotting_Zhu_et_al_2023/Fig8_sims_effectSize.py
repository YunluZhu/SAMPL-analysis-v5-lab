#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import get_figure_dir
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import set_font_type
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.stats import linregress
from sklearn.metrics import r2_score
from tqdm import tqdm

# %%
def sim_by_altered_coef(xdata, ydata, reg_func, coef_ori, coef_number_toAlter, change_ratio):
    coef_new = []
    for i, val in enumerate(coef_ori):
        if i == coef_number_toAlter:
            coef_new.append(val * (1-change_ratio))
        else:
            coef_new.append(val)

    yfit_ori = reg_func(xdata, *coef_ori)
    yfit_new = reg_func(xdata, *coef_new)

    yerrors = ydata - yfit_ori 
    ysim = yerrors + yfit_new
    return coef_new, ysim, yerrors

def calc_coef_effect_size(list_of_sample_N, xdata, ydata, sim_func, coef_ori_full, coef_number_toAlter, kinetic_get_stats, fig_name='coef effect size by percent chg'):
    """simulate data based on the coef to alter, perform regression on the new dataset

    Args:
        xdata (series): x values from real dataset
        ydata (series): y values from real dataset
        sim_func (func): model for regression
        list_of_sample_N (list): list of sample number to sample from the ori dataset
        kinetic_get_stats (func): func specific to each kinetic calculation. returns coef values and sigma
        coef_ori_full (list): a list of original coef
        coef_number_toAlter (int): index of the coef to alter

    Returns:
        dataframe: _description_
    """
    list_of_change_ratio = [0.01, 0.02, 0.05, 0.1]
    # list_of_change_ratio = [0.05]
    num_of_sampling_repeats = 200
    num_of_es_calculation_repeats = 20

    df_tocalc = pd.DataFrame()
    for change_ratio in tqdm(list_of_change_ratio):
        coef_new_imperial, ysim, yerrors = sim_by_altered_coef(xdata, ydata, sim_func, coef_ori_full, coef_number_toAlter, change_ratio)
        # xdata, ydata, ysim
        ori_data = pd.DataFrame(data={
            'x':xdata,
            'y':ydata
        })
        sim_data = pd.DataFrame(data={
            'x':xdata,
            'y':ysim
        })
        for sample_N in list_of_sample_N:
            for sampling_rep in np.arange(num_of_sampling_repeats):
                for es_cal_rep in np.arange(num_of_es_calculation_repeats):
                    # bootstrap sampling
                    ori_sample = ori_data.iloc[np.random.randint(sample_N, size=sample_N)]
                    sim_sample = sim_data.iloc[np.random.randint(sample_N, size=sample_N)]
                    res = kinetic_get_stats(ori_sample, sim_sample)
                    res = res.assign(
                        sampling_rep = sampling_rep,
                        cal_rep = es_cal_rep,
                        sample_number = sample_N,
                        percent_diff = change_ratio*100,
                    )
                    df_tocalc = pd.concat([df_tocalc,res],ignore_index=True)
                
    # use {num_of_sampling_repeats} to calculate ES, repeat for {num_of_es_calculation_repeats} times to get mean ES           
    df_toplt = df_tocalc.groupby(['sample_number','percent_diff','cal_rep']).mean()
    df_tocalc["id"] = df_tocalc.index
    df_combined_long = pd.melt(df_tocalc, id_vars=['id'], value_vars=['value_ori', 'value_sim']).merge(df_tocalc, how='left', on='id')
    combined_std = df_combined_long.groupby(['sample_number','percent_diff'])[['value']].std()
    df_toplt = df_toplt.assign(
        effect_size = (df_toplt['value_ori'] - df_toplt['value_sim']) / combined_std['value']
    )
    df_toplt = df_toplt.reset_index()
    
    plt.figure(figsize=(4,3))
    g = sns.lineplot(
        data = df_toplt,
        x = 'sample_number',
        y = 'effect_size',
        hue = 'percent_diff',
        legend='full',
        errorbar = 'sd'
    )
    g.set(xscale='log')
    # if df_toplt['effect_size'].max() < 3:
    #     if df_toplt['effect_size'].max() < 1:
    #         g.set(xlim=[0,None])
    #     else:
    #         g.set(xlim=[0,1])
    # else:
    #     g.set(xlim=[0,2])

    ci_fig = get_figure_dir('Fig_8')
    
    sns.despine()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    g.set(title=fig_name)
    filename = os.path.join(ci_fig,fig_name+".pdf")
    plt.savefig(filename,format='PDF')
    return df_toplt

def plot_estimation_error(df_tocalc, nobs_col = 'nobs', value_col = 'values', sigma_col = 'sigma', fig_name = 'Bout Number vs coef effect size'):
    # percentChg_list = np.array([0.01, 0.02, 0.05, 0.1])
    res = pd.DataFrame()
    # for percent_chg in percentChg_list:
    est_error = []
    nobs_list = []
    for nobs, group in df_tocalc.groupby(nobs_col):
        mu = group[value_col].mean()
        sigma = group[sigma_col].mean()
        est_error.append(sigma)
        nobs_list.append(nobs)
    to_concat = pd.DataFrame (data={
        'est_error': est_error,
        'bout number': nobs_list,
        # 'percent_chg': percent_chg,
        })
    res = pd.concat([res,to_concat],ignore_index=True)
    plt.figure(figsize=(4,3))
    g = sns.lineplot(
        data = res,
        x = 'est_error',
        y = 'bout_number',
        # hue = 'percent_chg',
        legend='full'
    )

    ci_fig = get_figure_dir('Fig_8')

    sns.despine()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    g.set(title=fig_name)
    filename = os.path.join(ci_fig,fig_name+".pdf")
    plt.savefig(filename,format='PDF')
    return res

def parabola_func(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def linear_func(x, a, b):
    return a*x+b

def parabola_fit(df, **kwargs):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    X_RANGE_to_fit = [0]
    for key, value in kwargs.items():
        if key == 'x_to_fit':
            X_RANGE_to_fit = value
            
            
    ydata = df['bout_freq']
    xdata = df['propBoutIEI_pitch']
    popt, pcov = curve_fit(parabola_func, xdata, ydata, 
                           p0=(0.005,3,0.5) , 
                           bounds=((0, -5, 0),(10, 15, 10)))
    y = []
    for x in X_RANGE_to_fit:
        y.append(parabola_func(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    r_squared = r2_score(ydata, parabola_func(xdata, *popt))
    return output_coef, output_fitted, p_sigma, r_squared

def sigmoid_fit2(df,func,**kwargs):
    x_range_to_fit = [0]
    for key, value in kwargs.items():
        if key == 'x_to_fit':
            x_range_to_fit = value
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
    r_squared = r2_score(ydata, func(xdata, *popt))   
    return output_coef, output_fitted, p_sigma, r_squared

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

def finBodyRatio_get_stats(ori_sample, sim_sample):
    dataset_dict = {
        'ori': ori_sample,
        'sim': sim_sample
    }
    res_dict = {}
    for key, data in dataset_dict.items():
        data.columns = ['rot_to_max_angvel','atk_ang']
        coef, _, sigma, _ = sigmoid_fit2(
            data, func=sigfunc_4free
        )
        E_height = coef.iloc[0,3]
        E_k = coef.iloc[0,0] 
        # V_height = sigma[3]**2  # height error
        # V_k = sigma[0]**2  # k error
        # slope_var = (V_height*V_k + V_height*(E_k**2) +  V_k*(E_height**2)) * (1/4)**2
        slope = E_k * E_height / 4
        res_dict['value_'+key] = slope
        # res_dict['sigma_'+key] = np.sqrt(slope_var)
    res = pd.DataFrame(data=res_dict, index=[0])
    return res

def sensitivity_get_stats(ori_sample, sim_sample):
    dataset_dict = {
        'ori': ori_sample,
        'sim': sim_sample
    }
    res_dict = {}
    for key, data in dataset_dict.items():
        data.columns = ['propBoutIEI_pitch','bout_freq']
        coef_master, _, sigma_master, r_squared_master = parabola_fit(data)
        res_dict['value_'+key] = coef_master.iloc[0,0]*1000
        # res_dict['sigma_'+key] = sigma_master[0]*1000
    res = pd.DataFrame(data=res_dict, index=[0])
    return res

def steeringGain_get_stats(ori_sample, sim_sample):
    dataset_dict = {
        'ori': ori_sample,
        'sim': sim_sample
    }
    res_dict = {}
    for key, data in dataset_dict.items():
        xdata = data['x']
        ydata = data['y']
        slope, _, _, _, std_err = linregress(xdata, ydata)
        res_dict['value_'+key] = slope
        # res_dict['sigma_'+key] = std_err
    res = pd.DataFrame(data=res_dict, index=[0])
    return res

def rightingGain_get_stats(ori_sample, sim_sample):
    dataset_dict = {
        'ori': ori_sample,
        'sim': sim_sample
    }
    res_dict = {}
    for key, data in dataset_dict.items():
        xdata = data['x']
        ydata = data['y']
        slope, _, _, _, std_err = linregress(xdata, ydata)
        res_dict['value_'+key] = slope * (-1)
        # res_dict['sigma_'+key] = std_err
    res = pd.DataFrame(data=res_dict, index=[0])
    return res

def Fig8_sims_effectSize(root):
    """ Estimate effect size

    Steps:
        1. for a given percentage of difference, simulate dataset using sim_by_altered_coef()
        2. sample desired number of data points from original and simulated dataset
        3. do regression, on sampled ori data, sampled sim data, repeat #2, #3 1000 times
        4. achieve distribution of parameter_ori, parameter_simulated
        5. calculate cohen's d as effect size
    """

    # Select data and create figure folder
    
    set_font_type()
    mpl.rc('figure', max_open_warning = 0)

    which_ztime = 'day'
    FRAME_RATE = 166
    
    ci_fig = get_figure_dir('Fig_8')

    try:
        os.makedirs(ci_fig)
    except:
        pass

    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

    number_of_N = 6

    print("- Figure 8: sensitivity effect size")
    IBI_angles.dropna(inplace=True)

    # def IBI_timging_sim_by_sensitivityChg(df, which_coef, diff_ratio):
    coef_ori_full, _,_, _ = parabola_fit(IBI_angles)
    coef_ori_full = coef_ori_full.values.flatten()

    coef_number_toAlter = 0
    xdata = IBI_angles['propBoutIEI_pitch']
    ydata = IBI_angles['bout_freq']

    list_of_sample_N = np.logspace(np.log10(200),np.log10(len(IBI_angles)//1000*1000),number_of_N).astype(int)
    # list_of_sample_N = np.linspace(200,5000,8).astype(int)
    es = calc_coef_effect_size(list_of_sample_N, xdata, ydata, parabola_func, coef_ori_full, coef_number_toAlter, sensitivity_get_stats, fig_name = 'Sensitivity')

    print("- Figure 8: steering gain effect size")

    xcol = 'traj_peak'
    ycol = 'pitch_peak'
    xdata = all_feature_cond[xcol] 
    ydata = all_feature_cond[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    coef_ori_full = [slope, intercept]
    coef_number_toAlter = 0

    list_of_sample_N = np.logspace(np.log10(200),np.log10(len(all_feature_cond)//1000*1000),number_of_N).astype(int)
    es = calc_coef_effect_size(list_of_sample_N, xdata, ydata, linear_func, coef_ori_full, coef_number_toAlter, steeringGain_get_stats, fig_name = 'Steering gain')


    print("- Figure 8: righting gain effect size")

    xcol = 'pitch_initial'
    ycol = 'rot_righting'
    xdata = all_feature_cond[xcol] 
    ydata = all_feature_cond[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    coef_ori_full = [slope, intercept]
    coef_number_toAlter = 0
    
    # list_of_sample_N = np.linspace(200,len(all_feature_cond)//1000*1000,number_of_N).astype(int)
    list_of_sample_N = np.logspace(np.log10(200),np.log10(len(all_feature_cond)//1000*1000),number_of_N).astype(int)
    es = calc_coef_effect_size(list_of_sample_N, xdata, ydata, linear_func, coef_ori_full, coef_number_toAlter, rightingGain_get_stats, fig_name = 'Righting gain')


    print("- Figure 8: fin-body ratio effect size")
    bouts_to_plot = all_feature_cond.loc[all_feature_cond['spd_peak']>=7]
    bouts_to_plot = bouts_to_plot.drop(bouts_to_plot.loc[(bouts_to_plot['atk_ang']<0) & (bouts_to_plot['rot_steering']>all_feature_cond['rot_steering'].median())].index)
    
    coef_ori_full, _, sigma, r_squared = sigmoid_fit2(
        bouts_to_plot, func=sigfunc_4free
    )
    E_height = coef_ori_full.iloc[0,3]
    E_k = coef_ori_full.iloc[0,0] 
    V_height = sigma[3]**2  # height error
    V_k = sigma[0]**2  # k error
    slope_var = (V_height*V_k + V_height*(E_k**2) +  V_k*(E_height**2)) * (1/4)**2  # estimate slope error
    slope = E_k * E_height / 4

    coef_number_toAlter = 0 # change k, keep height the same
    xdata = bouts_to_plot['rot_to_max_angvel']
    ydata = bouts_to_plot['atk_ang']
    coef_ori_full = coef_ori_full.values.flatten()
    
    # list_of_sample_N = np.linspace(2000,len(bouts_to_plot)//1000*1000,number_of_N).astype(int)
    list_of_sample_N = np.logspace(np.log10(2000),np.log10(len(bouts_to_plot)//1000*1000),number_of_N).astype(int)

    es = calc_coef_effect_size(list_of_sample_N, xdata, ydata, sigfunc_4free, coef_ori_full, coef_number_toAlter, finBodyRatio_get_stats, fig_name = 'Fin-body ratio')

    
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig8_sims_effectSize(root)