#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_IBIangles import get_IBIangles
# import scipy.stats as st
from sklearn.metrics import r2_score


def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-25,40+bin_width,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','bout_freq']].mean()
    return df_out
    
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

def Fig3_bout_timing(root):
    set_font_type()
    defaultPlotting()
    # %%
    which_ztime = 'day'
    FRAME_RATE = 166
    SAMPLE_N = 0

    folder_dir = os.getcwd()
    folder_name = f'Bout timing'
    folder_dir3 = get_figure_dir('Fig_3')
    fig_dir = os.path.join(folder_dir3, folder_name)

    try:
        os.makedirs(fig_dir)
    except:
        pass

    print('- Figure 3: Bout timing')

    # %%
    # CONSTANTS
    # SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
    BIN_WIDTH = 3 # this adjusts the density of raw data points on the fitted parabola
    X_RANGE_FULL = range(-30,41,1)
    frequency_th = 3 / 40 * FRAME_RATE

    # %%
    IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])
    IBI_angles['expNum'] = IBI_angles.groupby(['condition0','condition','expNum']).ngroup()
    IBI_angles['condition0'] = '7DD'
    IBI_angles['condition'] = 'WT'

    # %% distribution of IBI
    toplt = IBI_angles
    all_features = ['propBoutIEI_pitch','propBoutIEI']
    for feature_toplt in (all_features):
        # let's add unit
        if 'pitch' in feature_toplt:
            xlabel = "IBI pitch (deg)"
        else:
            xlabel = "Inter-bout interval (s)"
        plt.figure(figsize=(3,2))
        upper = np.nanpercentile(toplt[feature_toplt], 99.5)
        lower = np.nanpercentile(toplt[feature_toplt], 0.5)
        
        g = sns.histplot(data=toplt, x=feature_toplt, 
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
        plt.savefig(fig_dir+f"/{feature_toplt} distribution.pdf",format='PDF')



    # %%
    # Distribution plot, bout frequency vs IBI pitch

    toplt = IBI_angles[['bout_freq','propBoutIEI_pitch']]
    toplt.columns=['Bout frequency (Hz)','IBI pitch (deg)']
    plt.figure()
    g = sns.displot(data=toplt,
                    x='IBI pitch (deg)',y='Bout frequency (Hz)',
                    aspect=0.8,
                    cbar=True)
    g.set(xlim=(-25, 45),
        ylim=(None,3.5)
        )

    filename = os.path.join(fig_dir,f"IBI timing distribution.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    coef_rep = pd.DataFrame()
    y_rep = pd.DataFrame()
    binned_angles = pd.DataFrame()
    cat_cols = ['condition','condition0','ztime']

    IBI_sampled = IBI_angles
    if SAMPLE_N !=0:
        IBI_sampled = IBI_sampled.groupby(['condition','condition0','ztime','exp']).sample(
            n=SAMPLE_N,
            replace=True,
            )
        
    for (this_cond, this_condition0, this_ztime), group in IBI_sampled.groupby(cat_cols):
        rep_list = group['expNum'].unique()        
        for expNum in rep_list:
            this_df_toFit = group.loc[group['expNum']==expNum,['propBoutIEI_pitch','bout_freq','propBoutIEI']].reset_index(drop=True)
            this_df_toFit.dropna(inplace=True)
            coef, fitted_y, p_sigma = parabola_fit1(this_df_toFit, X_RANGE_FULL)
            coef_rep = pd.concat([coef_rep, coef.assign(condition0=this_condition0,
                                                                    condition=this_cond,
                                                                    expNum=expNum,
                                                                    ztime=this_ztime)])
            y_rep = pd.concat([y_rep, fitted_y.assign(condition0=this_condition0,
                                                                    condition=this_cond,
                                                                    expNum=expNum,
                                                                    ztime=this_ztime)])
            
        this_binned_angles = distribution_binned_average(group, BIN_WIDTH)
        this_binned_angles = this_binned_angles.assign(condition0=this_condition0,
                                                        condition=this_cond,
                                                        ztime=this_ztime)
        binned_angles = pd.concat([binned_angles, this_binned_angles],ignore_index=True)

    y_rep.columns = ['Bout frequency','IBI pitch','condition0','condition','repeat_num','ztime']
    y_rep = y_rep.reset_index(drop=True)
    coef_columns = ['Sensitivity (mHz/deg^2)','Baseline posture (deg)','Base bout rate (Hz)']
    coef_names = ['Sensitivity','Baseline posture','Base bout rate']
    coef_rep.columns = coef_columns + ['condition0','condition','repeat_num','ztime']
    coef_rep = coef_rep.reset_index(drop=True)

    binned_angles = binned_angles.reset_index(drop=True)

    all_ztime = list(set(coef_rep['ztime']))
    all_ztime.sort()
    ori_mean_coef = coef_rep[coef_columns].mean().values
    coef_rep['Sensitivity (mHz/deg^2)'] = coef_rep['Sensitivity (mHz/deg^2)']*1000

    print("Mean of coefficients for fitted parabola:")
    mean_coef = coef_rep[coef_columns].mean()
    print(mean_coef)
    print("Std. of coefficients for fitted parabola:")
    print(coef_rep[coef_columns].std())

    IBI_sampled.dropna(inplace=True)
    r_squared = r2_score(IBI_sampled['bout_freq'], ffunc1(IBI_sampled['propBoutIEI_pitch'], *ori_mean_coef))
    print(f"r-squared of the parabola modeling: {r_squared:3f}")
    # %% plot bout frequency vs IBI pitch, fit with parabola

    g = sns.relplot(x='IBI pitch',y='Bout frequency', data=y_rep, 
                    kind='line',
                    col='condition0',
                    hue='condition',errorbar='sd',
                    aspect=0.8
                    )
    g.map(sns.scatterplot,
          data=binned_angles,
          x='propBoutIEI_pitch', y='bout_freq', 
          hue='condition',alpha=0.2)
    g.set(xlim=(-25, 45),
        ylim=(None,2)
        )
    g._legend.remove()
        
    filename = os.path.join(fig_dir,f"IBI timing parabola fit.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    # plot all coefs

    plt.close()

    for i, coef_col_name in enumerate(coef_columns):
        p = sns.catplot(
            data = coef_rep, y=coef_col_name,x='condition0',kind='point',join=False,errorbar='sd',
            hue='condition', dodge=True,sharey=False
        )
        p.map(sns.lineplot,'condition0',coef_col_name,estimator=None,
            units='repeat_num',
            hue='condition',
            alpha=0.2,
            data=coef_rep)
        filename = os.path.join(fig_dir,f"IBI {coef_names[i]} ± SD.pdf")
        plt.savefig(filename,format='PDF')

if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig3_bout_timing(root)