#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, plot_pointplt)
from plot_functions.get_IBIangles import get_IBIangles
from statsmodels.stats.multicomp import MultiComparison


def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
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
    return output_coef, output_fitted

    # %%
def Fig7_bkg_timing(root):
    
    set_font_type()
    defaultPlotting()
    # %%
    which_ztime = 'day'
    SAMPLE_N = 0
    FRAME_RATE = 166
    
    folder_dir = os.getcwd()
    folder_name = f'bout_timing'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
    except:
        pass

    print('- Figure 7: Bout timing')

    # %%
    # CONSTANTS
    # SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
    BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
    X_RANGE_FULL = range(-30,41,1)
    frequency_th = 3 / 40 * FRAME_RATE
    IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

    # %%
    coef_val = pd.DataFrame()
    y_val = pd.DataFrame()
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
            coef, fitted_y = parabola_fit1(this_df_toFit, X_RANGE_FULL)
            coef_val = pd.concat([coef_val, coef.assign(condition0=this_condition0,
                                                                    condition=this_cond,
                                                                    expNum=expNum,
                                                                    ztime=this_ztime)])
            y_val = pd.concat([y_val, fitted_y.assign(condition0=this_condition0,
                                                                    condition=this_cond,
                                                                    expNum=expNum,
                                                                    ztime=this_ztime)])
            
        this_binned_angles = distribution_binned_average(group, BIN_WIDTH)
        this_binned_angles = this_binned_angles.assign(condition0=this_condition0,
                                                        condition=this_cond,
                                                        ztime=this_ztime)
        binned_angles = pd.concat([binned_angles, this_binned_angles],ignore_index=True)

    y_val.columns = ['Bout frequency (Hz)','IBI pitch (deg)','condition0','condition','repNum','ztime']
    y_val = y_val.reset_index(drop=True)
    coef_columns = ['Sensitivity (mHz/deg^2)','Baseline posture (deg)','Base bout rate (Hz)']
    coef_names = ['Sensitivity','Baseline posture','Base bout rate']
    coef_val.columns = coef_columns + ['condition0','condition','repNum','ztime']
    coef_val = coef_val.reset_index(drop=True)

    binned_angles = binned_angles.reset_index(drop=True)

    all_ztime = list(set(coef_val['ztime']))
    all_ztime.sort()

    coef_val['Sensitivity (mHz/deg^2)'] = coef_val['Sensitivity (mHz/deg^2)']*1000

    print("Mean of coefficients for fitted parabola:")
    print(coef_val[coef_columns].mean())
    print("Std. of coefficients for fitted parabola:")
    print(coef_val[coef_columns].std())
    # %% plot bout frequency vs IBI pitch, fit with parabola

    g = sns.relplot(x='IBI pitch (deg)',y='Bout frequency (Hz)', data=y_val, 
                    kind='line',
                    col='condition0', col_order=cond1_all,
                    hue='condition', hue_order = cond2_all,errorbar='sd',
                    aspect=0.9 , height = 3,
                    )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.scatterplot(data=binned_angles.loc[
                (binned_angles['condition0']==cond1_all[j]) & (binned_angles['ztime']==all_ztime[i]),:
                    ], 
                        x='propBoutIEI_pitch', y='bout_freq', 
                        hue='condition',alpha=0.2,legend=False,
                        ax=ax)
    g.set(xlim=(-25, 45),
        ylim=(None,2)
        )
    leg = g._legend
    leg.set_bbox_to_anchor([1,0.7])
    filename = os.path.join(fig_dir,f"IBI timing parabola fit.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    # plot all coefs

    plt.close()
    # %%
    # plot all coef
    for i, coef_col_name in enumerate(coef_columns):
        plot_pointplt(coef_val, coef_col_name, cond2_all)
        filename = os.path.join(fig_dir,f"IBI {coef_names[i]}.pdf")
        plt.savefig(filename,format='PDF')
        
        print(f"{coef_col_name}: ")
        multi_comp = MultiComparison(coef_val[coef_col_name], coef_val['condition'])
        print(multi_comp.tukeyhsd().summary())

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')  
    Fig7_bkg_timing(root)