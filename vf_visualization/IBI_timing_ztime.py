'''
This version loads "prop_Bout_IEI2" from IEI_data.h5 and reads 'propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime'
conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcdefghi in this case)
age info is taken from the 3 characters in folder names (xAA in this case, does not need to be number) – UPDATED 210618
AA represents light dark conditions (LD or DD or LL...) or whatever other condiitions. Does not need to be number
 
outputs: 
    plots of fiitted parabola (jackknifed, FULL parabola), to make half parabola, run script parabola_sensitivity_half.py
    plots of fiitted coefs of function y = a * ((x-b)**2) + c (jackknifed)
    plots of paired sensitivities (jackknifed)
    paired T test results for sensitivities if number of conditions per age == 2
    multiple comparison results for sensitiivity if number of conditions per age > 2

NOTE: bounds in parabola_fit1() confines the upper and lower limites of the parameters. These may need to be changed according to your own data sets.
'''

#%%
import os
from this import d
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_IBIangles import get_IBIangles

set_font_type()
# %%
pick_data = 'tau_long'
which_ztime = 'all'

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI_timing_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
# CONSTANTS
# SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
X_RANGE_FULL = range(-40,41,1)

def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    # df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean()
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit = X_RANGE_FULL):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['y_boutFreq'], 
                           p0=(0.005,3,0.5) , 
                           bounds=((0, -10, 0),(10, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %%
IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(y_boutFreq=1/IBI_angles['propBoutIEI'])
# %%
jackknifed_coef = pd.DataFrame()
jackknifed_y = pd.DataFrame()
binned_angles = pd.DataFrame()
cat_cols = ['condition','dpf','ztime']
for (this_cond, this_dpf, this_ztime), group in IBI_angles.groupby(cat_cols):
    jackknife_idx = jackknife_resampling(np.array(list(range(IBI_angles['expNum'].max()+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_df_toFit = group.loc[group['expNum'].isin(idx_group),['propBoutIEI_pitch','y_boutFreq','propBoutIEI']].reset_index(drop=True)
        this_df_toFit.dropna(inplace=True)
        coef, fitted_y = parabola_fit1(this_df_toFit, X_RANGE_FULL)
        jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=this_dpf,
                                                                condition=this_cond,
                                                                excluded_exp=excluded_exp,
                                                                ztime=this_ztime)])
        jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=this_dpf,
                                                                condition=this_cond,
                                                                excluded_exp=excluded_exp,
                                                                ztime=this_ztime)])
        
    this_binned_angles = distribution_binned_average(this_df_toFit, BIN_WIDTH)
    this_binned_angles = this_binned_angles.assign(dpf=this_dpf,
                                                    condition=this_cond,
                                                    ztime=this_ztime)
    binned_angles = pd.concat([binned_angles, this_binned_angles],ignore_index=True)

jackknifed_y.columns = ['bout frequency','IBI pitch','dpf','condition','jackknife num','ztime']
jackknifed_y = jackknifed_y.reset_index(drop=True)

jackknifed_coef.columns = ['sensitivity','x intersect','y intersect','dpf','condition','jackknife num','ztime']
jackknifed_coef = jackknifed_coef.reset_index(drop=True)

binned_angles = binned_angles.reset_index(drop=True)

all_ztime = list(set(jackknifed_coef['ztime']))
all_ztime.sort()
# %% plot
g = sns.relplot(x='IBI pitch',y='bout frequency', data=jackknifed_y, 
                kind='line',
                col='dpf', col_order=cond1_all,
                row = 'ztime', row_order=all_ztime,
                hue='condition', hue_order = cond2_all,ci='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.scatterplot(data=binned_angles.loc[
            (binned_angles['dpf']==cond1_all[j]) & (binned_angles['ztime']==all_ztime[i]),:
                ], 
                    x='propBoutIEI_pitch', y='y_boutFreq', 
                    hue='condition',alpha=0.2,
                    ax=ax)
g.set(xlim=(-40, 40),
      ylim=(0,4))
    
filename = os.path.join(fig_dir,"IEI timing.pdf")
plt.savefig(filename,format='PDF')
plt.show()

# %%
# plot slope
plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
for i in np.arange(len(coef.columns)):
    p = sns.catplot(
        data = jackknifed_coef, y=col_to_plt[i],x='dpf',kind='point',join=False,
        col_order=cond1_all,ci='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='condition', dodge=True,
        hue_order = cond2_all,sharey=False
    
    )
    p.map(sns.lineplot,'dpf',col_to_plt[i],estimator=None,
        units='jackknife num',
        hue='condition',
        alpha=0.2,
        data=jackknifed_coef)
    filename = os.path.join(fig_dir,f"IBI coef {i}.pdf")
    plt.savefig(filename,format='PDF')

plt.show()

# %%
