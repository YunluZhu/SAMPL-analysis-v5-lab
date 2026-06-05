'''
plot mean IBI bout frequency vs. IBI pitch and fit with a parabola
UP DN separated

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - ONE sample number for day and night
- change the var RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change it to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_list)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_functions import plt_categorical_grid2


##### Parameters to change #####
pick_data = 'sldp' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
if_jackknife = True # whether to calculate std of IBI pitch on jackknife dataset or for individual repeats
RESAMPLE = 0 # same resample number applied to day and night
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI2_timing_byBox_z{which_ztime}_sample{RESAMPLE}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')
set_font_type()
defaultPlotting()
# %%
# CONSTANTS
# SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
X_RANGE_FULL = range(-30,51,1)
frequency_th = 3 / 40 * FRAME_RATE
if if_jackknife:
    prename = 'jackknife'
else:
    prename = ''

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
                           p0=(0.005,-1,0.28) , 
                           bounds=((0, -10, 0),(10, 25, 10)))
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    return output_coef, output_fitted

# %%
IBI_angles, cond0_all, cond1_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(y_boutFreq=1/IBI_angles['propBoutIEI'])
IBI_angles = IBI_angles.loc[IBI_angles['y_boutFreq']<frequency_th]
# IBI_angles = IBI_angles.loc[IBI_angles['propBoutIEI_angVel'].abs()<30]
# IBI_angles = IBI_angles.loc[IBI_angles['propBoutIEI_pitch'].abs()<65]
IBI_angles = IBI_angles.assign(
    expBoxNum = IBI_angles['expNum'].astype(str) + '_' + IBI_angles['boxNum'].astype(str) 
)
# %%
resampled_coef = pd.DataFrame()

for rep in np.arange(100):
    cat_cols = ['cond1','cond0']

    IBI_sampled = IBI_angles
    if RESAMPLE !=0:
        IBI_sampled = IBI_sampled.groupby(['cond1','cond0','expBoxNum']).sample(
            n=RESAMPLE,
            replace=True,
            )
    for (this_cond1, this_cond0), group in IBI_sampled.groupby(cat_cols):
        if if_jackknife:
            jackknife_idx = jackknife_list(group['expBoxNum'].unique())            
        else:
            jackknife_idx = [[i] for i in group['expBoxNum'].unique()]
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_df_toFit = group.loc[group['expBoxNum'].isin(idx_group),['propBoutIEI_pitch','y_boutFreq','propBoutIEI']].reset_index(drop=True)
            this_df_toFit.dropna(inplace=True)
            coef, fitted_y = parabola_fit1(this_df_toFit, X_RANGE_FULL)
            resampled_coef = pd.concat([resampled_coef, coef.assign(cond0=this_cond0,
                                                                    cond1=this_cond1,
                                                                    excluded_exp=excluded_exp,
                                                                    rep = rep)],
                                       ignore_index=True)
            
resampled_coef.columns = ['sensitivity','x intersect','y intersect','cond0','cond1','resample num','rep']
resampled_coef = resampled_coef.reset_index(drop=True)
    
resampled_coef['sensitivity'] = resampled_coef['sensitivity']*1000

# %% 
####################################
###### Plotting Starts Here ######
####################################



for feature in ['sensitivity', 'x intersect', 'y intersect']:
    x_name = 'cond1'
    gridrow = None
    gridcol = 'cond0'
    g = plt_categorical_grid2(
        data = resampled_coef,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = 'resample num',
        sharey=False,
        height = 3,
        aspect = 0.8,
        )
    filename = os.path.join(fig_dir,f"{prename}_{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
for (cond0, rep), this_rep_coef in resampled_coef.groupby(['cond0', 'rep']):
    condition2_df = this_rep_coef.loc[this_rep_coef['cond1'] == cond1_all[2]]
    condition_df = this_rep_coef.loc[this_rep_coef['cond1'] == cond1_all[1]]
    control_df = this_rep_coef.loc[this_rep_coef['cond1'] == cond1_all[0]]
    _, p = ttest_rel(control_df, condition_df)
# %%
