#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,jackknife_list)
from plot_functions.get_bout_kinetics import get_bout_kinetics
import matplotlib as mpl
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel
from lmfit import Model

set_font_type()

def parabola_func(x, a, b, c, **kwargs):
    # parabola function
    return a*((x-b)**2)+c

def parabola_func_log(x, a, b, c):
    # parabola function
    return np.log10(a*((x-b)**2)+c)

def parabola_reg(df, yname, xname, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    df = df.sort_values(by=xname)
    df = df[[xname, yname]].dropna()
    mod  = Model(parabola_func)
    res = mod.fit(df[yname],x=df[xname],a=0.001, b=1, c=5)
    parameters = res.params.valuesdict()
    pars = parameters.values()
    
    y = [parabola_func(x_val, *pars) for x_val in X_RANGE_to_fit]
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    
    y_est = []
    for x in df[xname].values:
        y_est.append(parabola_func(x,*pars))
    
    r2 = r2_score(df[yname].values, y_est)
    _ = None
    return pd.DataFrame(data=parameters, index=[0]), output_fitted, _, r2

def parabola_reg_log(df, yname, xname, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    df = df.sort_values(by=xname)
    df = df[[xname, yname]].dropna()
    mod  = Model(parabola_func_log)
    res = mod.fit(np.log10(df[yname]),x=df[xname],a=0.001, b=1, c=5)
    parameters = res.params.valuesdict()
    pars = parameters.values()
    
    y = [parabola_func(x_val, *pars) for x_val in X_RANGE_to_fit]
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    
    y_est = []
    for x in df[xname].values:
        y_est.append(parabola_func(x,*pars))
    
    r2 = r2_score(df[yname].values, y_est)
    _ = None
    return pd.DataFrame(data=parameters, index=[0]), output_fitted, _, r2


# %%
pick_data = 'sldp' # name of your cond0 to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
if_jackknife = False

##### Parameters to change #####

# %%

root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'BF5_spdMod_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')


root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_ztime)

# %%

all_feature_cond = all_feature_cond.assign(
    pitch_peak_abs = all_feature_cond['pitch_peak'].abs(),
    yspd_peak = all_feature_cond['spd_peak']*np.sin(all_feature_cond['traj_peak']* np.pi / 180),
    xspd_peak = all_feature_cond['spd_peak']*np.cos(all_feature_cond['traj_peak']* np.pi / 180),
    # lift_ratio = all_feature_cond['lift_distance_fullBout']/all_feature_cond['depth_chg_fullBout'],
)
all_feature_cond = all_feature_cond.assign(
    yposture_spd_peak = all_feature_cond['xspd_peak']*np.tan(all_feature_cond['pitch_peak']* np.pi / 180),
)
all_feature_cond = all_feature_cond.assign(
    lift_spd = all_feature_cond['yspd_peak'] - all_feature_cond['yposture_spd_peak']
)

all_feature_cond = all_feature_cond.assign(
    atk_ang_bins = pd.cut(all_feature_cond.atk_ang, bins=[-50,-20,-2,3,20,50]),
    spd_peak_adj = all_feature_cond['spd_peak']
)


# %%  fit with parabola, use lmfit module
df_toplt = all_feature_cond
# df_toplt = df_toplt.query("spd_peak_adj > 5")

x_name = 'pitch_peak'
y_name = 'spd_peak'
data = df_toplt
# x_range = np.arange(-35,40,0.5)

reg_res = pd.DataFrame()
model_res = pd.DataFrame()
binned_angles = pd.DataFrame()
for (cond0, cond1), this_df in data.groupby(['cond0','cond1']):  
    x_range_min = np.percentile(this_df[x_name],1)
    x_range_max = np.percentile(this_df[x_name],99)
    this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
    x_range = np.arange(x_range_min, x_range_max, 1)
    
    exp_df = this_df.groupby('expNum').size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]
        
    # jackknife_exp_matrix = jackknife_list(this_df['expNum'].unique())
    
    for j, exp_group in enumerate(jackknife_exp_matrix):
        rep_group = this_df.loc[this_df['expNum'].isin(exp_group),:]
        
        y_var = np.var(rep_group[y_name])
        x_var = np.var(rep_group[x_name])
        y75 = np.percentile(rep_group[y_name], 75)
        y25 = np.percentile(rep_group[y_name], 25)
        ystd = np.std(rep_group[y_name])

        
        output_coef, output_fitted, p_sigma, r2 = parabola_reg_log(
            df=rep_group,
            xname=x_name,
            yname=y_name,
            X_RANGE_to_fit=x_range
        )
        output_coef.columns=['k', 'b', 'c']
        output_fitted.columns=['yfit','xrange']
        output_coef = output_coef.assign(
            y_var = y_var,
            x_var = x_var,
            y75 = y75,
            y25 = y25,
            ystd = ystd,
            # k_sigma=p_sigma[0],
            # b_sigma=p_sigma[1],
            # c_sigma=p_sigma[2],
            r_square=r2,
            cond0=cond0,
            cond1=cond1,
            expNum=j,
        )
        output_fitted = output_fitted.assign(
            cond0=cond0,
            cond1=cond1,
            expNum=j
        )
        
        this_binned_angles = distribution_binned_average(
            df=rep_group,
            by_col=x_name,
            bin_col=y_name,
            bin=x_range).assign(
                cond0=cond0,
                cond1=cond1,
                expNum=j
            )  

        reg_res = pd.concat([reg_res,output_coef], ignore_index=True)
        model_res = pd.concat([model_res,output_fitted], ignore_index=True)
        binned_angles = pd.concat([binned_angles, this_binned_angles], ignore_index=True)

reg_res = reg_res.assign(
    y_intersect = reg_res['b']**2 * reg_res['k'] + reg_res['c'],
)
# reg_res = reg_res.assign(
#     wtf = np.sqrt(reg_res['k'] / reg_res['y_intersect']),
#     wtf2 = reg_res['y_intersect'] / reg_res['y_var'] * reg_res['k'] / reg_res['x_var'],
#     wtf3 = reg_res['y_intersect'] / np.sqrt(reg_res['y_var']),
#     wtf4 = reg_res['k'] / reg_res['y75'],
#     wtf5 = np.sqrt(reg_res['k'] / reg_res['ystd']) / reg_res['y_intersect'] * reg_res['x_var'],
#     wtf6 = reg_res['k'] / reg_res['ystd']) / reg_res['y_intersect'] * reg_res['x_var'],
# )
# %% plot

x_name = x_name
y_name = y_name

g = sns.relplot(
    data=model_res,
    x='xrange',
    y='yfit',
    color='black',
    row='cond0',
    hue='cond1',
    # hue_order=['1sibs','2tau'],
    kind='line',
    height=3,
    # errorbar=get_2sd
)
for (row_val), ax in g.axes_dict.items():
    # sns.scatterplot(data=thiscond_data.sample(2000), x=x_name, y=y_name, ax=ax, alpha=0.07,
    #                 hue='cond1',
    #                 hue_order=['1sibs','2tau'],
    # )
    thiscond_data = binned_angles.query("cond0==@row_val")
    sns.lineplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, alpha=0.2,
                    hue='cond1', 
                    # hue_order=['1sibs','2tau'],
                    units='expNum', estimator=None
    )
    ax.legend([],[], frameon=False)


g.set(
    xlim=[np.percentile(data[x_name],1), np.percentile(data[x_name],99)],
    ylim=[np.percentile(data[y_name],0), np.percentile(data[y_name],99)],
)   

plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}.pdf",format='PDF')

# % coef
for i, coef_col_name in enumerate(['k','y_intersect','b']):
    p = sns.catplot(
        data = reg_res, y=coef_col_name,x='cond1', 
        col='cond0', 
        kind='point',join=False,errorbar='sd',
        hue='cond1',
        # 
        # hue_order=['1sibs','2tau'], 
        dodge=False, sharey=False, height=3, aspect=0.6
    )
    p.map(sns.lineplot,'cond1',coef_col_name,
          estimator=None,
          units='expNum',
          color='grey',
          alpha=0.2,
          data=reg_res)
    plt.savefig(fig_dir+f"/speed_modulation_reg_{coef_col_name}.pdf",format='PDF')




# %%
x_name = x_name
y_name = y_name


df_toplt = all_feature_cond#.query("atk_ang < -2 | atk_ang > 2")
# df_toplt = df_toplt.query("spd_peak_adj > 5")

x_name = 'pitch_peak'
y_name = 'spd_peak'
data=df_toplt
# x_range = np.arange(-35,40,0.5)


reg_res = reg_res.assign(
    y_intersect = reg_res['b']**2 * reg_res['k'] + reg_res['c'],
)

# %%
x_name = x_name
y_name = y_name

g = sns.relplot(
    data=model_res,
    x='xrange',
    y='yfit',
    color='black',
    row='cond0',
    col='cond1',
    hue='cond1',
    # hue_order=['1sibs','2tau'],
    kind='line',
    height=3,
    # errorbar=get_2sd
)
for (row_val, col_val), ax in g.axes_dict.items():
    thiscond_data = data.query("cond0==@row_val & cond1==@col_val")
    sns.histplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
                    cbar=True, pthresh=.05,
                    hue='cond1', 
                    # hue_order=['1sibs','2tau'], 
                    stat='count',
                    palette = 'gray'
    )
    thiscond_data = binned_angles.query("cond0==@row_val & cond1==@col_val")
    sns.lineplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, alpha=0.2,
                    hue='cond1', 
                    # hue_order=['1sibs','2tau'],
                    units='expNum', estimator=None, 
    )
    ax.legend([],[], frameon=False)


g.set(
    xlim=[np.percentile(data[x_name],1), np.percentile(data[x_name],99)],
    ylim=[np.percentile(data[y_name],0), np.percentile(data[y_name],99)],
    xticks=[-40, 0, 40]
)   


plt.savefig(fig_dir+f"/speed_modulation_distribution.pdf",format='PDF')


# #%%
# g = sns.relplot(
#     data=master_binned_angles,
#     x=x_name,
#     y=y_name,
#     color='white',
#     row='cond0',
#     col='cond1',
#     kind='line',
#     height=3
# )
# for (row_val, col_val), ax in g.axes_dict.items():
#     thiscond_data = data.query("cond0==@row_val & cond1==@col_val")
#     sns.histplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
#                     cbar=True, pthresh=.05,
#                     hue='cond1', hue_order=['1sibs','2tau'], stat='count',
#                     palette = 'gray'
#     )
#     ax.legend([],[], frameon=False)


# g.set(
#     xlim=[np.percentile(data[x_name],1), np.percentile(data[x_name],99)],
#     ylim=[np.percentile(data[y_name],0), np.percentile(data[y_name],99)],
# )   
# plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}_histogram.pdf",format='PDF')

# %% no fitting, plot raw per repeat

# %%  fit with parabola, use lmfit module
df_toplt = all_feature_cond#.loc[np.abs(all_feature_cond['rot_total']) <10]
# df_toplt = df_toplt.query("spd_peak_adj > 5")

x_name = 'pitch_peak'
y_name = 'spd_peak'

if_jackknife = False
# x_range = np.arange(-35,40,0.5)

binned_angles_raw = pd.DataFrame()

for (cond0, cond1), this_df in df_toplt.groupby(['cond0','cond1']):  
    x_range_min = np.percentile(this_df[x_name],1)
    x_range_max = np.percentile(this_df[x_name],99)
    this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
    x_range = np.arange(x_range_min, x_range_max, 1)
    
    exp_df = this_df.groupby('expNum').size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]
        
    # jackknife_exp_matrix = jackknife_list(this_df['expNum'].unique())
    for j, exp_group in enumerate(jackknife_exp_matrix):
        rep_group = this_df.loc[this_df['expNum'].isin(exp_group),:]

        
        this_binned_angles = distribution_binned_average(
            df=rep_group,
            by_col=x_name,
            bin_col=y_name,
            bin=x_range).assign(
                cond0=cond0,
                cond1=cond1,
                expNum=j
            )  

        # reg_res = pd.concat([reg_res,output_coef], ignore_index=True)
        # model_res = pd.concat([model_res,output_fitted], ignore_index=True)
        binned_angles_raw = pd.concat([binned_angles_raw, this_binned_angles], ignore_index=True)



g = sns.relplot(
    data=binned_angles_raw,
    x=x_name,
    y=y_name,
    row='cond1',
    hue='expNum',
    # hue_order=['1sibs','2tau'],
    kind='line',
    height=3,
    alpha=0.4,
    units='expNum',
    estimator=None

)
g.set(
    # xlim=[np.percentile(binned_angles_raw[x_name],2), np.percentile(binned_angles_raw[x_name],98)],
    # ylim=[np.percentile(binned_angles_raw[y_name],1), np.percentile(binned_angles_raw[y_name],99)],
    xticks=[-40, 0, 40]
)  
plt.savefig(fig_dir+f"/speed modulation_individual repeats.pdf",format='PDF')

# %%

# %%