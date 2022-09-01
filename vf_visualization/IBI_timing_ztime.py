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
defaultPlotting()
# %%
pick_data = 'tau_long'
which_ztime = 'all'

RESAMPLE = 1000  # how many bouts to take per  exp/ztime/condition

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI_timing_z{which_ztime}_sample{RESAMPLE}'
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
X_RANGE_FULL = range(-30,51,1)

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
IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(y_boutFreq=1/IBI_angles['propBoutIEI'])
# %%

jackknifed_coef = pd.DataFrame()
jackknifed_y = pd.DataFrame()
binned_angles = pd.DataFrame()
cat_cols = ['condition','dpf','ztime']

IBI_sampled = IBI_angles
if RESAMPLE !=0:
    IBI_sampled = IBI_sampled.groupby(['condition','dpf','ztime','exp']).sample(
        n=RESAMPLE,
        replace=True,
        )
for (this_cond, this_dpf, this_ztime), group in IBI_sampled.groupby(cat_cols):
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

jackknifed_coef['sensitivity'] = jackknifed_coef['sensitivity']*1000

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
g.set(xlim=(-30, 50),
      ylim=(0,4))
    
filename = os.path.join(fig_dir,f"IEI timing sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')

# %%
# plot all coef

plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
for i in np.arange(len(coef.columns)):
    p = sns.catplot(
        data = jackknifed_coef, y=col_to_plt[i],x='dpf',kind='point',join=False,
        col_order=cond1_all,ci='sd',
        row = 'ztime', row_order=all_ztime,
        hue='condition', dodge=True,
        hue_order = cond2_all,sharey=False
    
    )
    p.map(sns.lineplot,'dpf',col_to_plt[i],estimator=None,
        units='jackknife num',
        hue='condition',
        alpha=0.2,
        data=jackknifed_coef)
    filename = os.path.join(fig_dir,f"IBI coef{i} sample{RESAMPLE}.pdf")
    plt.savefig(filename,format='PDF')

# %%
# plot all coef compare day night
plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
for i in np.arange(len(coef.columns)):
    p = sns.catplot(
        data = jackknifed_coef, y=col_to_plt[i],x='condition',
        kind='point',join=False,
        col='dpf',
        ci='sd',
        hue = 'ztime',
        # units=excluded_exp,
        dodge=True,
        # hue_order = cond1_all,
        sharey=False
    )
    p.map(sns.lineplot,'condition',col_to_plt[i],estimator=None,
        units='jackknife num',
        hue='ztime',
        alpha=0.2,
        data=jackknifed_coef)
    filename = os.path.join(fig_dir,f"IBI coef{i} day-night sample{RESAMPLE}.pdf")
    plt.savefig(filename,format='PDF')

# %%
# plot sensitivity
plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
p = sns.catplot(
    data = jackknifed_coef, y='sensitivity',x='condition',
    kind='point',join=False,
    col='dpf', col_order=cond1_all,
    ci='sd',
    row = 'ztime', row_order=all_ztime,
    hue='condition', 
    hue_order = cond2_all,sharey=True,
    aspect=.8, 
    # markers=['d','d'],
)
p.map(sns.lineplot,'condition','sensitivity',estimator=None,
    units='jackknife num',
    color='grey',
    alpha=0.2,
    data=jackknifed_coef)
filename = os.path.join(fig_dir,f"IBI sensitivity sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')

# %%
# plot x intersect
plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
p = sns.catplot(
    data = jackknifed_coef, y='x intersect',x='condition',
    kind='point',join=False,
    col='dpf', col_order=cond1_all,
    ci='sd',
    row = 'ztime', row_order=all_ztime,
    hue='condition',
    hue_order = cond2_all,sharey=True,
    aspect=.8, 
    # markers=['d','d']

)
p.map(sns.lineplot,'condition','x intersect',estimator=None,
    units='jackknife num',
    color='grey',
    alpha=0.2,
    data=jackknifed_coef)
filename = os.path.join(fig_dir,f"IBI base posture sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')

# %%
# plot baseline bout rate
plt.close()
col_to_plt = {0:'sensitivity',1:'x intersect',2:'y intersect'}
p = sns.catplot(
    data = jackknifed_coef, y='y intersect',x='condition',
    kind='point',join=False,
    col='dpf', col_order=cond1_all,
    ci='sd',
    row = 'ztime', row_order=all_ztime,
    hue='condition',
    hue_order = cond2_all,sharey=True,
    aspect=.8, 
    # markers=['d','d']

)
p.map(sns.lineplot,'condition','y intersect',estimator=None,
    units='jackknife num',
    color='grey',
    alpha=0.2,
    data=jackknifed_coef)
filename = os.path.join(fig_dir,f"IBI baseline rate sample{RESAMPLE}.pdf")
plt.savefig(filename,format='PDF')
# %%
# test code

plt.close()
p = sns.catplot(
    data = jackknifed_coef, y='sensitivity',x='condition',
    kind='point',
    col='dpf',
    ci='sd',
    row = 'ztime', 
    hue='jackknife num', 
    # hue_order = cond2_all,
    sharey=True,
    aspect=.8, 
    # markers=['d','d'],
)
# p.map(sns.lineplot,'condition','sensitivity',
#     #   estimator=None,
#     # units='jackknife num',
#     hue='jackknife num',
#     alpha=0.2,
#     data=jackknifed_coef)


# %%
