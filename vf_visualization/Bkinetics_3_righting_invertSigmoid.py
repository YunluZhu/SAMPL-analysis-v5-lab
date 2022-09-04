'''

'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean, set_font_type, defaultPlotting, distribution_binned_average)

set_font_type()
# defaultPlotting()
# %%
pick_data = 'tau_long'
which_zeitgeber = 'day'
DAY_RESAMPLE = 1000
# %%
def sigmoid_fit(x_val, y_val, x_range_to_fit,func,**kwargs):
    # lower_bounds = [0.1,-20,-100,1]
    # upper_bounds = [5,20,2,100]
    x0=[1, 0.5, 20, -30]
    
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
    popt, pcov = curve_fit(func,x_val, y_val, 
                        #    maxfev=2000, 
                           p0 = p0,
                        #    bounds=(lower_bounds,upper_bounds)
                        )
    y = func(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y


# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-5,8.01,0.01)
BIN_WIDTH = 0.3
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK3_righting_invertSigFit_z{which_zeitgeber}_sample{DAY_RESAMPLE}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)

# %% fit sigmoid - master
all_coef = pd.DataFrame()
all_y = pd.DataFrame()
all_binned_average = pd.DataFrame()

df_tofit = all_feature_cond.loc[all_feature_cond['spd_peak']>6,:]

if DAY_RESAMPLE != 0:
    df_tofit = all_feature_cond.groupby(
            ['dpf','condition','expNum']
            ).sample(
                    n=DAY_RESAMPLE,
                    replace=True
                    )
for (cond_abla,cond_dpf), for_fit in df_tofit.groupby(['condition','dpf']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        this_for_fit = for_fit.loc[for_fit['expNum'].isin(idx_group)]
        coef, fitted_y, sigma = sigmoid_fit(
            this_for_fit['rot_l_decel'],this_for_fit['pitch_pre_bout'], X_RANGE, func=sigfunc_4free
        )
        slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
        fitted_y.columns = ['pre_bout pitch','decel rotation']
        all_y = pd.concat([all_y, fitted_y.assign(
            dpf=cond_dpf,
            condition=cond_abla,
            excluded_exp = excluded_exp,
            ztime=which_zeitgeber,
            )])
        all_coef = pd.concat([all_coef, coef.assign(
            slope=slope,
            dpf=cond_dpf,
            condition=cond_abla,
            excluded_exp = excluded_exp,
            ztime=which_zeitgeber,
            )])
    binned_df = distribution_binned_average(for_fit,by_col='rot_l_decel',bin_col='pitch_pre_bout',bin=AVERAGE_BIN)
    binned_df.columns=['decel rotation','pre_bout pitch']
    all_binned_average = pd.concat([all_binned_average,binned_df.assign(
        dpf=cond_dpf,
        condition=cond_abla,
        ztime=which_zeitgeber,
        )],ignore_index=True)
    
all_y = all_y.reset_index(drop=True)
all_coef = all_coef.reset_index(drop=True)

all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()
# %%
# plot sigmoid
plt.figure()

g = sns.relplot(x='decel rotation',y='pre_bout pitch', data=all_y, 
                kind='line',
                col='dpf', col_order=all_cond1,
                row = 'ztime', row_order=all_ztime,
                hue='condition', hue_order = all_cond2,ci='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=all_binned_average.loc[
            (all_binned_average['dpf']==all_cond1[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                ], 
                    x='decel rotation', y='pre_bout pitch', 
                    hue='condition',alpha=0.5,
                    ax=ax)
    
filename = os.path.join(fig_dir,"pre_bout pitch VS decel rotation.pdf")
plt.savefig(filename,format='PDF')

plt.figure()

g = sns.relplot(x='decel rotation',y='pre_bout pitch', data=all_y, 
                kind='line',
                col='condition', col_order=all_cond2,
                row = 'ztime', row_order=all_ztime,
                hue='dpf', hue_order = all_cond1, 
                ci='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=all_binned_average.loc[
            (all_binned_average['condition']==all_cond2[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                ], 
                    x='decel rotation', y='pre_bout pitch', 
                    hue='dpf',alpha=0.5,
                    ax=ax)
    
filename = os.path.join(fig_dir,"pre_bout pitch VS decel rotation by cond1.pdf")
plt.savefig(filename,format='PDF')
# %%
# plot coefs
plt.close()
all_coef.columns = ['a','b','c','d','slope','dpf','condition','excluded_exp','ztime']
all_coef_comp = all_coef.assign(
    upper = all_coef['c'].values,
    lower = all_coef['c'].values + all_coef['d'].values,
    x_off = all_coef['b'].values,
    growth = all_coef['a'].values,
    gain = -1/(all_coef['slope'])
)
    
for feature in ['upper','lower','x_off','growth','gain','slope']:
    p = sns.catplot(
        data = all_coef_comp, y=feature,x='dpf',kind='point',join=False,
        col_order=all_cond1,ci='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='condition', dodge=True,
        hue_order = all_cond2,
    )
    p.map(sns.lineplot,'dpf',feature,estimator=None,
        units='excluded_exp',
        hue='condition',
        alpha=0.2,
        data=all_coef_comp)
    filename = os.path.join(fig_dir,f"coef vs age {feature}.pdf")
    plt.savefig(filename,format='PDF')

# %%
for feature in ['upper','lower','x_off','growth','gain','slope']:
    p = sns.catplot(
        data = all_coef_comp, y=feature,x='condition',kind='point',join=False,
        col='dpf', col_order=all_cond1,ci='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='condition', 
        hue_order = all_cond2,
        aspect=0.2*len(all_cond2),
    )
    p.map(sns.lineplot,'condition',feature,estimator=None,
        units='excluded_exp',
        color='grey',
        alpha=0.2,
        data=all_coef_comp)
    filename = os.path.join(fig_dir,f"coef by age {feature}.pdf")
    plt.savefig(filename,format='PDF')
# %%
