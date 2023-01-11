#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.get_bout_features import get_bout_features,get_max_angvel_rot
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average_nostd)
from scipy import stats
import math
from sklearn.metrics import r2_score

# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-5,-100,1]
    upper_bounds = [15,20,2,100]
    x0=[3, 1, 0, 5]

    for key, value in kwargs.items():
        if key == 'a':
            x0[0] = value
            lower_bounds[0] = value-0.01
            upper_bounds[0] = value+0.01
        elif key == 'b':
            x0[1] = value
            lower_bounds[1] = value-0.01
            upper_bounds[1] = value+0.01
        elif key == 'c':
            x0[2] = value
            lower_bounds[2] = value-0.01
            upper_bounds[2] = value+0.01
        elif key =='d':
            x0[3] = value
            lower_bounds[3] = value-0.01
            upper_bounds[3] = value+0.01            
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
def Fig5_fin_body_coordination(root, root_fin):
    data_dir = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data"
    root = os.path.join(data_dir,'DD_7dpf')
    root_finless = os.path.join(data_dir,'DD_finless')
    # Fig5_fin_body_coordination(root, root_finless)

    ######################################################
    set_font_type()
    which_ztime = 'day'
    FRAME_RATE = 166


    folder_name = f'Atk_ang fin_body_ratio'
    folder_dir5 = get_figure_dir('Fig_5')
    fig_dir = os.path.join(folder_dir5, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure directory:')
        print(fig_dir)

    # %%
    # CONSTANTS
    X_RANGE = np.arange(-2.2,20,0.05)
    BIN_WIDTH = 0.5
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)
    # %%
    # get data
    max_angvel_time, _, _ = get_max_angvel_rot(root, FRAME_RATE)
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, max_angvel_time=max_angvel_time)
    # %%
    all_feature_cond = all_feature_cond.reset_index(drop=True)
    all_feature_cond['condition0'] = '7DD'
    all_feature_cond['condition'] = 'WT'
    # %%
    print("- Figure 5: Distribution of early body rotation and attack angle")
    feature_to_plt = ['rot_to_max_angvel','atk_ang']
    toplt = all_feature_cond

    for feature in feature_to_plt:
        plt.figure(figsize=(3,2))
        upper = np.percentile(toplt[feature], 99.5)
        lower = np.percentile(toplt[feature], 0.5)
        if feature == 'atk_ang':
            atk_ctrl_upper = upper
            atk_ctrl_lower = lower
        xlabel = feature + " (deg)"
        
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
        plt.savefig(fig_dir+f"/{feature} distribution before filtering.pdf",format='PDF')
        # plt.close()
    # %% tidy data
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

    sel_bouts = all_feature_cond.drop(all_feature_cond.loc[(all_feature_cond['atk_ang']<0) & (all_feature_cond['rot_steering']>all_feature_cond['rot_steering'].median())].index)
    sel_bouts = sel_bouts.query('spd_peak > 7')
    filtered_bouts = all_feature_cond.loc[~all_feature_cond.index.isin(sel_bouts.index)]

    all_feature_cond = all_feature_cond.assign(bout_filter = 'selected')
    all_feature_cond.loc[all_feature_cond.index.isin(filtered_bouts.index),'bout_filter'] = 'excluded'
    # %% 5D
    print("- Figure 5: correlation of attack angle with rotation and rotation residual")
    plt_dict = {
        'early_rotation vs atk_ang':['rot_to_max_angvel','atk_ang'],
        'late_rotation vs atk_ang':['rot_residual','atk_ang'],
    }

    # %%
    toplt = all_feature_cond

    for which_to_plot in plt_dict:
        [x,y] = plt_dict[which_to_plot]
        upper = np.percentile(toplt[x], 99)
        lower = np.percentile(toplt[x], 2)
        BIN_WIDTH = 0.5
        AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
        binned_df = sel_bouts.groupby(['condition','condition0']).apply(
            lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
        )
        binned_df.columns=[x,y]
        binned_df = binned_df.reset_index(level=['condition0','condition'])
        binned_df = binned_df.reset_index(drop=True)

        # xlabel = "Relative pitch change (deg)"
        # ylabel = 'Trajectory deviation (deg)'

        g = sns.relplot(
            kind='scatter',
            data = toplt.sample(n=3000),
            row='condition',
            col = 'condition0',
            # col_order = all_cond1,
            # row_order = all_cond2,
            x = x,
            y = y,
            alpha=0.1,
            linewidth = 0,
            hue = 'bout_filter',
            hue_order = ['selected','excluded'],
            height=3,
            aspect=2/2,
            legend=False
            )

        g.map(sns.lineplot,
            data = binned_df,
            x = x,
            y = y,
            ax=g.ax
            )
        
        g.set(ylim=(-12,16))
        g.set(xlim=(lower,upper))
        g.set(xlabel=x+" (deg)")
        g.set(ylabel=y+" (deg)")

        # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
        sns.despine()
        plt.savefig(fig_dir+f"/{x} {y} correlation.pdf",format='PDF')
        r_val = stats.pearsonr(toplt[x],toplt[y])[0]
        print(f"pearson's r = {r_val}")
        
    # %% fit sigmoid 
    df_toplt = sel_bouts.reset_index(drop=True)

    all_coef = pd.DataFrame()
    all_y = pd.DataFrame()
    all_binned_average = pd.DataFrame()

    upper = np.percentile(df_toplt['rot_to_max_angvel'], 99)
    lower = np.percentile(df_toplt['rot_to_max_angvel'], 1)
    BIN_WIDTH = 0.4
    AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)


    for (condition1,cond_condition0,cond_ztime), data_cond in df_toplt.groupby(['condition','condition0','ztime']):
        units = data_cond['expNum'].unique()
        for exp_rep in units:
            coef, fitted_y, sigma = sigmoid_fit(
                data_cond.loc[data_cond['expNum']==exp_rep], X_RANGE, func=sigfunc_4free,
                )
            slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
            fitted_y.columns = ['Attack angle (deg)','rotation (deg)']
            all_y = pd.concat([all_y, fitted_y.assign(
                condition0=cond_condition0,
                condition=condition1,
                excluded_exp = exp_rep,
                ztime=cond_ztime,
                )])
            all_coef = pd.concat([all_coef, coef.assign(
                slope=slope,
                condition0=cond_condition0,
                condition=condition1,
                excluded_exp = exp_rep,
                ztime=cond_ztime,
                )])
            
        fit_data = data_cond.drop(data_cond.loc[(data_cond['atk_ang']<0) & (data_cond['rot_steering']>5)].index)
        binned_df = distribution_binned_average_nostd(fit_data,by_col='rot_to_max_angvel',bin_col='atk_ang',bin=AVERAGE_BIN)
        binned_df.columns=['rotation (deg)','atk_ang']
        all_binned_average = pd.concat([all_binned_average,binned_df.assign(
            condition0=cond_condition0,
            condition=condition1,
            ztime=cond_ztime,
            )],ignore_index=True)
        
    all_y = all_y.reset_index(drop=True)
    all_coef = all_coef.reset_index(drop=True)
    all_coef.columns=['k','xval','min','height',
                    'slope','condition0','condition','excluded_exp','ztime']
    all_ztime = list(set(all_coef['ztime']))
    all_ztime.sort()

    defaultPlotting(size=12)

    plt.figure()

    g = sns.relplot(x='rotation (deg)',y='Attack angle (deg)', data=all_y, 
                    kind='line',
                    col='condition0', 
                    errorbar='sd',
                    )
    sns.lineplot(data=all_binned_average,
                        x='rotation (deg)', y='atk_ang', alpha=0.5,
                        ax=g.ax)
    # upper = np.percentile(df_toplt['atk_ang'], 80)
    # lower = np.percentile(df_toplt['atk_ang'], 20)
    g.set(ylim=(-2.5, 11))
    g.set(xlim=(-2.5, 20))

    filename = os.path.join(fig_dir,"fin-body coordination.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    # plot 
    # plt.close()
    defaultPlotting(size=12)
    plt.figure()
    p = sns.catplot(
        data = all_coef, y='slope',x='condition0',kind='point',join=False,errorbar='sd',
        hue_order = all_cond2,
    )
    p.map(sns.lineplot,'condition0','slope',estimator=None,
        units='excluded_exp',
        hue='condition',
        alpha=0.2,
        data=all_coef)
    filename = os.path.join(fig_dir,"slope_together.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    mean_val = all_coef['slope'].mean()
    std_val = all_coef['slope'].std()
    print(f"maximal slope: {mean_val:.3f}±{std_val:.3f}")

    mean_val = (all_coef['height']+all_coef['min']).mean()
    std_val = (all_coef['height']+all_coef['min']).std()
    print(f"upper asymptote: {mean_val:.3f}±{std_val:.3f}")
    
    mean_coef_forcalc = all_coef[['k','xval','min','height']].mean().values
    r_squared = r2_score(df_toplt['atk_ang'], sigfunc_4free(df_toplt['rot_to_max_angvel'], *mean_coef_forcalc))
    print(f"r-squared of the sigmoid modeling: {r_squared:3f}")
    # %%

    # plot finless fish data
    all_feature_finless, fin_cond1, fin_cond2 = get_bout_features(root_finless, FRAME_RATE)
    all_feature_finless = all_feature_finless.reset_index(drop=True)

    feature = 'atk_ang'
    if 'spd' in feature:
        xlabel = feature + " (mm*s^-1)"
    elif 'dis' in feature:
        xlabel = feature + " (mm)"
    else:
        xlabel = feature + " (deg)"
    plt.figure(figsize=(3,2))
    upper = atk_ctrl_upper
    lower =atk_ctrl_lower

    g = sns.histplot(data=all_feature_finless, 
                    x=feature, 
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
    plt.savefig(os.path.join(fig_dir,f"{feature} distribution finless fish.pdf"),format='PDF')
# %%
if __name__ == "__main__":
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    root_finless = os.path.join(data_dir,'DD_finless')
    Fig5_fin_body_coordination(root, root_finless)