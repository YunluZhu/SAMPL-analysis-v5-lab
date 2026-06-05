'''
Plot speed as a function of posture, fit with parabola
Jackknife or not
multi-dataset
Sample same number of data from each experimental repeat for each dataset
Curve fit based on filtered data
bin posture with width 1-2deg, for each bin, exclude speed data that are <25 and >75 percentile of speed in the given bin
This method yields a fitted line very close to raw values
Alternatively, getting a robust fit requires using log on Y for regression

'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,jackknife_list)
from plot_functions.get_bout_kinetics import get_bout_kinetics
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
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
    res = mod.fit(df[yname],x=df[xname],a=0.001, b=10, c=10)
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

# def parabola_reg_log(df, yname, xname, X_RANGE_to_fit):
#     '''
#     fit bout probability - pitch to parabola
#     May need to adjust bounds
#     '''
#     df = df.sort_values(by=xname)
#     df = df[[xname, yname]].dropna()
#     mod  = Model(parabola_func_log)
#     try:
#         res = mod.fit(np.log10(df[yname]),x=df[xname],a=0.001, b=1, c=10)
#     except:
#         res = mod.fit(np.log10(df[yname]),x=df[xname], nan_policy='omit')
#     parameters = res.params.valuesdict()
#     pars = parameters.values()
    
#     y = [parabola_func(x_val, *pars) for x_val in X_RANGE_to_fit]
#     output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    
#     y_est = []
#     for x in df[xname].values:
#         y_est.append(parabola_func(x,*pars))
    
#     r2 = r2_score(df[yname].values, y_est)
#     _ = None
#     return pd.DataFrame(data=parameters, index=[0]), output_fitted, _, r2


def distribution_binned_average(df, xname, yname, x_range):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by=xname)
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
    grp = df.groupby(bins)
    df_out = grp[[xname,yname]].median()
    return df_out

def distribution_binned_zscore(df, xname, yname, x_range):
    '''
    bins raw pitch data using fixed bin width. Return binned z of pitch and bout frequency.
    '''
    df = df.sort_values(by=xname)
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
    grp = df.groupby(bins)
    df_out = grp[[xname,yname]].median()
    return df_out

def thresh_filter_average(obj, sort_var, tgt_var, q=(0.1, 0.9), stat='median'):
    q1, q2 = q
    thresh1 = obj[sort_var].quantile(q=q1)
    thresh2 = obj[sort_var].quantile(q=q2)
    return getattr(obj[(obj[sort_var] >= thresh1) & (obj[sort_var] <= thresh2)][tgt_var], stat)()

def thresh_filter(obj, sort_var, q=(0.1, 0.9)):
    """Filter by percentile

    Args:
        obj (dataFrame): data to filter
        sort_var (str): column to filter by
        q (tuple, optional): lower and upper percentile threshold for filtering. Defaults to (0.1, 0.9).

    Returns:
        pd.DataFrame: _description_
    """
    q1, q2 = q
    thresh1 = obj[sort_var].quantile(q=q1)
    thresh2 = obj[sort_var].quantile(q=q2)
    return obj[(obj[sort_var] >= thresh1) & (obj[sort_var] <= thresh2)]


def distribution_binned_FilterByPercentile_mean(df, xname, yname, x_range):
    '''
    bins raw pitch data using fixed bin width. Return binned z of pitch and bout frequency.
    '''
    df = df.sort_values(by=xname)
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
    grp = df.groupby(bins)
    df_out = grp[[xname,yname]].apply(lambda x: thresh_filter_average(x, yname, yname)).to_frame()
    df_out.columns = [yname]
    df_out[xname] = grp[[xname]].median()
    return df_out

# %%

# data_list = ['otog','nMLF','nMLF_axon'] # all or specific data
data_list = ['otog','nMLF_axon','nMLF_small'] # all or specific data
data_name = ''.join(data_list)


folder_name = f'spdMod consecbouts'
folder_dir = get_figure_dir(data_name)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass


which_zeitgeber = 'day'
if_jackknife = True

df_features_combined = pd.DataFrame()

for pick_data in data_list:
    root, FRAME_RATE = get_data_dir(pick_data)
    all_feature_cond, _, _ = get_connected_bouts(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond['dataset'] = pick_data
    df_features_combined = pd.concat([df_features_combined,all_feature_cond], ignore_index=True)

df_features_combined = df_features_combined.assign(
    condition = df_features_combined['cond1'].map({'otog': "cond", 
                                                       'hets':'ctrl',
                                                       '1ctrl':'ctrl', 
                                                       '2cond': 'cond', 
                                                       'lesion':'cond',
                                                       'ctrl': 'ctrl',
                                                       'lesionD1':'ctrl', 
                                                       'lesionD2':'cond', 
                                                       'finless': 'cond'})
)


# %%

df_features_combined = df_features_combined.assign(
    pitch_peak_abs = df_features_combined['pitch_peak'].abs(),
    yspd_peak = df_features_combined['spd_peak']*np.sin(df_features_combined['traj_peak']* np.pi / 180),
    xspd_peak = df_features_combined['spd_peak']*np.cos(df_features_combined['traj_peak']* np.pi / 180),
    lift_ratio = df_features_combined['additional_depth_chg']/df_features_combined['depth_chg'],
)
df_features_combined = df_features_combined.assign(
    yposture_spd_peak = df_features_combined['xspd_peak']*np.tan(df_features_combined['pitch_peak']* np.pi / 180),
)
df_features_combined = df_features_combined.assign(
    lift_spd = df_features_combined['yspd_peak'] - df_features_combined['yposture_spd_peak']
)

df_features_combined = df_features_combined.assign(
    atk_ang_bins = pd.cut(df_features_combined.atk_ang, bins=[-50,-20,-2,3,20,50]),
)
# df_toplt = df_features_combined.query("lift_ratio > -1 & lift_ratio < 2")

# %%  fit with parabola, use lmfit module
df_toplt = df_features_combined

x_name = 'pitch_peak'
y_name = 'spd_peak'
if_jackknife = True

# x_range = np.arange(-35,40,0.5)


reg_res = pd.DataFrame()
model_res = pd.DataFrame()
binned_angles = pd.DataFrame()
for (dataset, condition), this_df_ori in df_toplt.groupby(['dataset','condition']):  
    sample_num_per_rep = int(this_df_ori.groupby(['expNum']).size().median())
    this_df = this_df_ori.groupby(['expNum']).sample(sample_num_per_rep, replace=True)

    x_range_min = -30#np.percentile(this_df[x_name],0.2)
    x_range_max = 40#np.percentile(this_df[x_name],99.8)
    this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
    x_range = np.arange(x_range_min, x_range_max, 1)
    
    exp_df = this_df.groupby('expNum').size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]
            
    for j, exp_group in enumerate(jackknife_exp_matrix):
        rep_raw = this_df.loc[this_df['expNum'].isin(exp_group),:]
        # filtering data
        
        df_tosort = rep_raw.sort_values(by=x_name)
        # df = df.assign(bout_freq = 1/df['propBoutIEI'])
        bins = pd.cut(df_tosort[x_name], list(np.arange(x_range_min,x_range_max,2)))
        grp = df_tosort.groupby(bins)
        df_out = grp[[x_name, y_name]].apply(lambda bined_vals: thresh_filter(bined_vals, y_name, q=(0.25,0.75))).reset_index(drop=True)
        # bin_names = df_out[x_name].unique()
        # df_out[x_name] = df_out[x_name].map(
        #     dict(
        #         zip(bin_names, grp[[x_name]].mean()[x_name].values)
        #     )
        # )
        #######
        reg_data = df_out # rep_group df_out
        # rep_group = df_out # rep_group df_out

        #######
        
        # y_var = np.var(rep_group[y_name])
        # x_var = np.var(rep_group[x_name])
        # y75 = np.percentile(rep_group[y_name], 75)
        # y25 = np.percentile(rep_group[y_name], 25)
        # ystd = np.std(rep_group[y_name])

        
        output_coef, output_fitted, p_sigma, r2 = parabola_reg(
            df=reg_data,
            xname=x_name,
            yname=y_name,
            X_RANGE_to_fit=x_range
        )
        output_coef.columns=['k', 'b', 'c']
        output_fitted.columns=['yfit','xrange']
        output_coef = output_coef.assign(
            # y_var = y_var,
            # x_var = x_var,
            # y75 = y75,
            # y25 = y25,
            # ystd = ystd,
            # k_sigma=p_sigma[0],
            # b_sigma=p_sigma[1],
            # c_sigma=p_sigma[2],
            r_square=r2,
            dataset=dataset,
            condition=condition,
            expNum=j,
        )
        output_fitted = output_fitted.assign(
            dataset=dataset,
            condition=condition,
            expNum=j
        )
        
        this_binned_angles = distribution_binned_average(
            df=rep_raw,
            xname=x_name,
            yname=y_name,
            x_range=x_range).assign(
                dataset=dataset,
                condition=condition,
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


#% plot

x_name = x_name
y_name = y_name

g = sns.relplot(
    data=model_res,
    x='xrange',
    y='yfit',
    color='black',
    row='dataset',
    hue='condition',
    hue_order=['ctrl','cond'],
    kind='line',
    height=3,
    # errorbar=get_2sd
    errorbar=None
)
for (row_val), ax in g.axes_dict.items():
    # sns.scatterplot(data=thiscond_data.sample(2000), x=x_name, y=y_name, ax=ax, alpha=0.07,
    #                 hue='condition',
    #                 hue_order=['ctrl','cond'],
    # )
    thiscond_data = binned_angles.query("dataset==@row_val")
    sns.lineplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, alpha=0.2,
                    hue='condition', 
                    hue_order=['ctrl','cond'],
                    units='expNum', estimator=None
    )
    ax.legend([],[], frameon=False)


g.set(
    xlim=[np.percentile(df_toplt[x_name],1), np.percentile(df_toplt[x_name],99)],
    ylim=[7, np.percentile(df_toplt[y_name],99)],
)   

plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}.pdf",format='PDF')

# % coef
# for i, coef_col_name in enumerate(['k','y_intersect','b']):
#     p = sns.catplot(
#         data = reg_res, y=coef_col_name,x='condition', 
#         col='dataset', 
#         kind='point',join=False,errorbar='sd',
#         hue='condition',hue_order=['ctrl','cond'], dodge=False, sharey=False, height=3, aspect=0.6
#     )
#     p.map(sns.lineplot,'condition',coef_col_name,
#           estimator=None,
#           units='expNum',
#           color='grey',
#           alpha=0.2,
#           data=reg_res)
#     plt.savefig(fig_dir+f"/speed_modulation_reg_{coef_col_name}.pdf",format='PDF')



# x_range = np.arange(-35,40,0.5)


reg_res = reg_res.assign(
    y_intersect = reg_res['b']**2 * reg_res['k'] + reg_res['c'],
)

# %
x_name = x_name
y_name = y_name

g = sns.relplot(
    data=model_res,
    x='xrange',
    y='yfit',
    color='black',
    row='dataset',
    col='condition',
    hue='condition',
    hue_order=['ctrl','cond'],
    kind='line',
    height=3,
    # errorbar=get_2sd
)
for (row_val, col_val), ax in g.axes_dict.items():
    thiscond_data = df_toplt.query("dataset==@row_val & condition==@col_val")
    sns.histplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
                    cbar=True, pthresh=.02,
                    hue='condition', hue_order=['ctrl','cond'], 
                    stat='count',
                    palette = 'gray'
    )
    thiscond_data = binned_angles.query("dataset==@row_val & condition==@col_val")
    sns.lineplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, alpha=0.2,
                    hue='condition', 
                    hue_order=['ctrl','cond'],
                    units='expNum', estimator=None, 
    )
    ax.legend([],[], frameon=False)


g.set(
    xlim=[np.percentile(df_toplt[x_name],1), np.percentile(df_toplt[x_name],99)],
    ylim=[7, np.percentile(df_toplt[y_name],99)],
    xticks=[-40, 0, 40]
)   


plt.savefig(fig_dir+f"/speed_modulation_distribution.pdf",format='PDF')


# #%%
# g = sns.relplot(
#     data=master_binned_angles,
#     x=x_name,
#     y=y_name,
#     color='white',
#     row='dataset',
#     col='condition',
#     kind='line',
#     height=3
# )
# for (row_val, col_val), ax in g.axes_dict.items():
#     thiscond_data = data.query("dataset==@row_val & condition==@col_val")
#     sns.histplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
#                     cbar=True, pthresh=.05,
#                     hue='condition', hue_order=['ctrl','cond'], stat='count',
#                     palette = 'gray'
#     )
#     ax.legend([],[], frameon=False)


# g.set(
#     xlim=[np.percentile(data[x_name],1), np.percentile(data[x_name],99)],
#     ylim=[np.percentile(data[y_name],0), np.percentile(data[y_name],99)],
# )   
# plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}_histogram.pdf",format='PDF')

# %% no fitting, plot raw per repeat

x_name = 'pitch_peak'
y_name = 'spd_peak'

# x_range = np.arange(-35,40,0.5)

df_toplt_filtered = df_toplt.dropna(subset=[x_name, y_name])
# df_toplt_filtered = df_toplt_filtered.query("pre_IBI_time > 4")

binned_angles_raw = pd.DataFrame()

g = sns.displot(
    df_toplt_filtered,
    x=x_name,
    y=y_name,
    col='condition',
    row='dataset',
    height=3,
    cbar=True, 
    hue='condition',
    pthresh=.02,
    palette='gray',
    stat='count',
)
g.set(
    xlim=[np.percentile(df_toplt_filtered[x_name],1), np.percentile(df_toplt_filtered[x_name],99)],
    ylim=[np.percentile(df_toplt_filtered[y_name],1), np.percentile(df_toplt_filtered[y_name],99)],
)   
# plt.savefig(fig_dir+f"/distribution.pdf",format='PDF')

###################

# %%

if_jackknife = True

for (dataset, condition), this_df in df_toplt_filtered.groupby(['dataset','condition']):  
    this_df = this_df.dropna(subset=[x_name, y_name])
    x_range_min = np.percentile(this_df[x_name],0.2)
    x_range_max = np.percentile(this_df[x_name],99.8)
    this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
    x_range = np.arange(x_range_min, x_range_max, (x_range_max-x_range_min)/100)
    
    exp_df = this_df.groupby('expNum').size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]
        
    # jackknife_exp_matrix = jackknife_list(this_df['expNum'].unique())
    for j, exp_group in enumerate(jackknife_exp_matrix):
        
        rep_group = this_df.loc[this_df['expNum'].isin(exp_group),:]

        
        # this_binned_angles = distribution_binned_average(
        this_binned_angles = distribution_binned_average(
            df=rep_group,
            xname=x_name,
            yname=y_name,
            x_range=x_range).assign(
                dataset=dataset,
                condition=condition,
                expNum=j,
                exp=''.join(rep_group.exp.unique())
            )  

        # reg_res = pd.concat([reg_res,output_coef], ignore_index=True)
        # model_res = pd.concat([model_res,output_fitted], ignore_index=True)
        binned_angles_raw = pd.concat([binned_angles_raw, this_binned_angles], ignore_index=True)



g = sns.relplot(
    data=binned_angles_raw,
    x=x_name,
    y=y_name,
    row='dataset',
    col='condition',
    hue='expNum',
    # palette=sns.color_palette("cubehelix", as_cmap=True),
    kind='line',
    height=3,
    alpha=0.4,
    units='exp',
    estimator=None,
    legend='full',

)
g.set(
    # xlim=[np.percentile(binned_angles_raw[x_name],2), np.percentile(binned_angles_raw[x_name],98)],
    # ylim=[np.percentile(binned_angles_raw[y_name],1), np.percentile(binned_angles_raw[y_name],99)],
    # xticks=[-40, 0, 40]
)  
plt.savefig(fig_dir+f"/speed modulation_individual repeats {y_name}.pdf",format='PDF')


 # %%

# %%