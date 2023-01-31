#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_stats import calc_ROC
from scipy import stats

pick_data = 'tau_bkg'
which_ztime = 'day'
DAY_RESAMPLE = 1000  # how many bouts to take per  exp/ztime/condition
RESAMPLE = DAY_RESAMPLE

set_font_type()
defaultPlotting(size=16)

# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-20,-100,1]
    upper_bounds = [5,20,2,100]
    x0=[0.1, 1, -1, 20]
    
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
    popt, pcov = curve_fit(func, df['rot_pre_bout'], df['atk_ang'], 
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

X_RANGE = np.arange(-5,10.01,0.01)
BIN_WIDTH = 0.3
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'stat_ROC'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime = which_ztime)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
if FRAME_RATE > 100:
    all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
elif FRAME_RATE == 40:
    all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<4].index, inplace=True)

# %% fit sigmoid - master
all_coef = pd.DataFrame()
all_y = pd.DataFrame()
all_binned_average = pd.DataFrame()


for (cond_abla,cond_dpf,cond_ztime), for_fit in all_feature_cond.groupby(['cond1','cond0','ztime']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        coef, fitted_y, sigma = sigmoid_fit(
            for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE, func=sigfunc_4free
        )
        slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
        fitted_y.columns = ['Attack angle','Pre-bout rotation']
        all_y = pd.concat([all_y, fitted_y.assign(
            dpf=cond_dpf,
            cond1=cond_abla,
            excluded_exp = excluded_exp,
            ztime=cond_ztime,
            )])
        all_coef = pd.concat([all_coef, coef.assign(
            slope=slope,
            dpf=cond_dpf,
            cond1=cond_abla,
            excluded_exp = excluded_exp,
            ztime=cond_ztime,
            )])
    binned_df = distribution_binned_average(for_fit,by_col='rot_pre_bout',bin_col='atk_ang',bin=AVERAGE_BIN)
    binned_df.columns=['Pre-bout rotation','atk_ang']
    all_binned_average = pd.concat([all_binned_average,binned_df.assign(
        dpf=cond_dpf,
        cond1=cond_abla,
        ztime=cond_ztime,
        )],ignore_index=True)
    
all_y = all_y.reset_index(drop=True)
all_coef = all_coef.reset_index(drop=True)
all_coef.columns=['k','xval','min','height',
                  'slope','cond0','cond1','excluded_exp','ztime']
all_ztime = list(set(all_coef['ztime']))
all_ztime.sort()

# %%
feature='height'
FPR_list, TPR_list, auc = calc_ROC(all_coef,feature,all_cond0[0],'increase')  
# TPR_list, FPR_list, auc = calc_ROC(jackknifed_coef,'x intersect',cond1_all[0],'right') 
# %%
fig, ax = plt.subplots(1,1, figsize=(3,3))

ax.plot(FPR_list, TPR_list)
ax.plot((0,1), "--")
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])
ax.set_title("ROC Curve", fontsize=14)
ax.set_ylabel('TPR', fontsize=12)
ax.set_xlabel('FPR', fontsize=12)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.legend([f"AUC = {np.mean(auc):.3f}±{np.std(auc):.3f}"])
filename = os.path.join(fig_dir,f"ROC_coordination_{feature}_{which_ztime}_sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')
# %%
for condition in all_cond0:
    df = all_coef.loc[all_coef.cond0 == condition]
    cond = df.loc[df.cond1==all_cond0[0],feature].values
    ctrl = df.loc[df.cond1==all_cond0[1],feature].values
    print(f'{condition} {feature} cond-ctrl paired ttest')
    print(stats.ttest_rel(cond,ctrl))

# %%
feature='slope'
TPR_list, FPR_list, auc = calc_ROC(all_coef,feature,all_cond0[0],'left')  # left = cond is expected to be smaller than ctrl
# TPR_list, FPR_list, auc = calc_ROC(jackknifed_coef,'x intersect',cond1_all[0],'right') 
# %%
fig, ax = plt.subplots(1,1, figsize=(3,3))

ax.plot(FPR_list, TPR_list)
ax.plot((0,1), "--")
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])
ax.set_title("ROC Curve", fontsize=14)
ax.set_ylabel('TPR', fontsize=12)
ax.set_xlabel('FPR', fontsize=12)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.legend([f"AUC = {np.mean(auc):.3f}±{np.std(auc):.3f}"])
filename = os.path.join(fig_dir,f"ROC_coordination_{feature}_{which_ztime}_sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')

for condition in all_cond0:
    df = all_coef.loc[all_coef.cond0 == condition]
    cond = df.loc[df.cond1==all_cond0[0],feature].values
    ctrl = df.loc[df.cond1==all_cond0[1],feature].values
    print(f'{condition} {feature} cond-ctrl paired ttest')
    print(stats.ttest_rel(cond,ctrl))