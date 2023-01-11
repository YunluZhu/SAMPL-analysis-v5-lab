#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.get_bout_features import get_max_angvel_rot, get_bout_features
from plot_functions.plt_tools import (set_font_type, defaultPlotting, plot_pointplt, distribution_binned_average_nostd)
from statsmodels.stats.multicomp import MultiComparison

    
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-5,-100,1]
    upper_bounds = [15,20,2,100]
    x0=[3, 1, 0, 5]
    # for key, value in kwargs.items():
    #     if key == 'a':
    #         x0[0] = value
    #         lower_bounds[0] = value-0.01
    #         upper_bounds[0] = value+0.01
    #     elif key == 'b':
    #         x0[1] = value
    #         lower_bounds[1] = value-0.01
    #         upper_bounds[1] = value+0.01
    #     elif key == 'c':
    #         x0[2] = value
    #         lower_bounds[2] = value-0.01
    #         upper_bounds[2] = value+0.01
    #     elif key =='d':
    #         x0[3] = value
    #         lower_bounds[3] = value-0.01
    #         upper_bounds[3] = value+0.01
            
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df['rot_to_max_angvel'], df['atk_ang'], 
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

def Fig7_bkg_fin_body(root):
    set_font_type()
    defaultPlotting(size=16)
    # %%
    which_zeitgeber = 'day'
    DAY_RESAMPLE = 0
    NIGHT_RESAMPLE = 500
    # Select data and create figure folder
    FRAME_RATE = 166
    
    X_RANGE = np.arange(-5,20.05,0.05)
    BIN_WIDTH = 0.5
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

    print("- Figure 7: ZF strains - Fin-body coordination")

    folder_name = f'fin-body coordination'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass

    # %% get max_angvel_time per condition
    which_rotation = 'rot_to_max_angvel'
    which_atk_ang = 'atk_ang' # atk_ang or 'atk_ang_phased'
    # get features

    max_angvel_time, all_cond1, all_cond2 = get_max_angvel_rot(root, FRAME_RATE, ztime = which_zeitgeber)
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber, max_angvel_time = max_angvel_time)

    # %% tidy data
    df_toplt = pd.DataFrame()
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    for condition, grp in all_feature_cond.groupby('condition'):
        grp = grp.drop(grp.loc[(grp['atk_ang']<0) & (grp['rot_steering']>grp['rot_steering'].median())].index)
        grp = grp.query('spd_peak >= 7')
        df_toplt = pd.concat([df_toplt,grp])
    
    # %%
    angles_day_resampled = pd.DataFrame()
    angles_night_resampled = pd.DataFrame()

    if which_zeitgeber != 'night':
        angles_day_resampled = df_toplt.loc[
            df_toplt['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            angles_day_resampled = angles_day_resampled.groupby(
                    ['condition0','condition','expNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True,
                            # random_state=2
                            )
    if which_zeitgeber != 'day':
        angles_night_resampled = df_toplt.loc[
            df_toplt['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            angles_night_resampled = angles_night_resampled.groupby(
                    ['condition0','condition','expNum']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True,
                            # random_state=2
                            )
    df_toplt = pd.concat([angles_day_resampled,angles_night_resampled],ignore_index=True)
    
    # %% fit sigmoid 
    all_coef = pd.DataFrame()
    all_y = pd.DataFrame()
    all_binned_average = pd.DataFrame()

    for (condition,cond_condition0,cond_ztime), for_fit in df_toplt.groupby(['condition','condition0','ztime']):
        # expNum = for_fit['expNum'].max()
        rep_list = for_fit['expNum'].unique()
        for expNum in rep_list:
            coef, fitted_y, sigma = sigmoid_fit(
                for_fit.loc[for_fit['expNum'] == expNum], X_RANGE, func=sigfunc_4free
            )
            slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
            fitted_y.columns = ['Attack angle (deg)','Rotation (deg)']
            all_y = pd.concat([all_y, fitted_y.assign(
                condition0=cond_condition0,
                condition=condition,
                expNum = expNum,
                ztime=cond_ztime,
                )])
            all_coef = pd.concat([all_coef, coef.assign(
                slope=slope,
                condition0=cond_condition0,
                condition=condition,
                expNum = expNum,
                ztime=cond_ztime,
                )])
        binned_df = distribution_binned_average_nostd(for_fit,by_col=which_rotation,bin_col=which_atk_ang,bin=AVERAGE_BIN)
        binned_df.columns=['Rotation (deg)',which_atk_ang]
        all_binned_average = pd.concat([all_binned_average,binned_df.assign(
            condition0=cond_condition0,
            condition=condition,
            ztime=cond_ztime,
            )],ignore_index=True)
        
    all_y = all_y.reset_index(drop=True)
    all_coef = all_coef.reset_index(drop=True)
    all_coef.columns=['k','xval','min','height','slope','condition0','condition','expNum','ztime']
    all_ztime = list(set(all_coef['ztime']))
    all_ztime.sort()
    # %%
    defaultPlotting(size=12)

    plt.figure()

    g = sns.relplot(x='Rotation (deg)',y='Attack angle (deg)', data=all_y, 
                    kind='line',
                    hue='condition', hue_order = all_cond2, errorbar='sd',
                    )
    sns.lineplot(data=all_binned_average,
                x='Rotation (deg)', y=which_atk_ang, 
                hue='condition',alpha=0.5,legend=False,
                ax=g.ax)
    upper = np.percentile(df_toplt[which_atk_ang], 90)
    # lower = np.percentile(df_toplt[which_atk_ang], 10)
    g.set(ylim=(-4, upper))
    g.set(xlim=(-2.5, 20))

    filename = os.path.join(fig_dir,"attack angle vs rot to angvel max.pdf")
    plt.savefig(filename,format='PDF')

    # plot coef
    for coef_name in ['k','xval','min','height','slope']:
        plot_pointplt(all_coef,coef_name,all_cond2)
        filename = os.path.join(fig_dir,f"{coef_name}.pdf")
        plt.savefig(filename,format='PDF')
        
    
    # multiple comparison
    
    multi_comp = MultiComparison(all_coef['slope'], all_coef['condition'])
    print(multi_comp.tukeyhsd().summary())
    multi_comp = MultiComparison(all_coef['height'], all_coef['condition'])
    print(multi_comp.tukeyhsd().summary())
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig7_bkg_fin_body(root)