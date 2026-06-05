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
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average, get_2sd,jackknife_list)
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
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

def distribution_binned_average(df, xname, yname, bins):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by=xname)
    bins = pd.cut(df[xname], bins=list(bins))
    grp = df.groupby(bins, observed=False)
    df_out = grp[[xname,yname]].median()
    return df_out

# %%
folder_dir = get_figure_dir('Fig_3')
fig_dir = os.path.join(folder_dir, 'speedXposture_fit_')

# data_list = ['otog','nMLF','nMLF_axon'] # all or specific data
pick_data = 'tau_long'
            #  'nMLF','nMLF_small'] # all or specific data
fig_dir = fig_dir + pick_data

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass


which_zeitgeber = 'day'

df_features_combined = pd.DataFrame()

root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond['dataset'] = pick_data
df_features_combined = all_feature_cond.copy()

# df_features_combined = df_features_combined.assign(
#     cond1 = df_features_combined['cond1'].map({'otog': "2cond", 
#                                                 'hets':'1ctrl',
#                                                 '1ctrl':'1ctrl', 
#                                                 'cond': '2cond', 
#                                                 'lesion':'2cond',
#                                                 '1ctrl': '1ctrl',
#                                                 'lesionD1':'1ctrl', 
#                                                 'lesionD2':'2cond', 
#                                                 'finless': '2cond'})
# )
# %%

df_features_combined = df_features_combined.assign(
    pitch_peak_abs = df_features_combined['pitch_peak'].abs(),
    yspd_peak = df_features_combined['spd_peak']*np.sin(df_features_combined['traj_peak']* np.pi / 180),
    xspd_peak = df_features_combined['spd_peak']*np.cos(df_features_combined['traj_peak']* np.pi / 180),
    lift_ratio = df_features_combined['lift_distance_fullBout']/df_features_combined['depth_chg_fullBout'],
)
df_features_combined = df_features_combined.assign(
    yposture_spd_peak = df_features_combined['xspd_peak']*np.tan(df_features_combined['pitch_peak']* np.pi / 180),
)
df_features_combined = df_features_combined.assign(
    lift_spd = df_features_combined['yspd_peak'] - df_features_combined['yposture_spd_peak']
)

df_features_combined = df_features_combined.assign(
    atk_ang_bins = pd.cut(df_features_combined.atk_ang, bins=[-50,-20,-2,3,20,50]),
    spd_peak_adj = df_features_combined['spd_peak']
)

df_toplt = df_features_combined.query("spd_peak > 6")


#%%
x_name = 'atk_ang'
y_name = 'spd_peak'

reg_res = pd.DataFrame()
model_res = pd.DataFrame()
binned_angles = pd.DataFrame()
for (dataset, cond0, cond1), this_df_ori in df_toplt.groupby(['dataset','cond0','cond1'], observed=True):  
    sample_num_per_rep = int(this_df_ori.groupby(['expNum']).size().median())
    this_df = this_df_ori.groupby(['expNum']).sample(sample_num_per_rep, replace=True)

    x_range_min = -30#np.percentile(this_df[x_name],0.2)
    x_range_max = 45#np.percentile(this_df[x_name],99.8)
    this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
    x_range = np.arange(x_range_min, x_range_max, 5)
    
    exp_df = this_df.groupby('expNum').size()

    jackknife_exp_matrix = [[item] for item in exp_df.index]
            
    for j, exp_group in enumerate(jackknife_exp_matrix):
        rep_raw = this_df.loc[this_df['expNum'].isin(exp_group),:]
        # filtering data
        
        df_tosort = rep_raw.sort_values(by=x_name)
        # df = df.assign(bout_freq = 1/df['propBoutIEI'])
        bins = pd.cut(df_tosort[x_name], list(np.arange(x_range_min,x_range_max,2)))
        grp = df_tosort.groupby(bin, observed=True)
        df_out = grp[[x_name, y_name]].apply(lambda bined_vals: thresh_filter(bined_vals, y_name, q=(0.25,0.75))).reset_index(drop=True)

        #######
        reg_data = df_out # rep_group df_out

        #######
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
            cond1=cond1,
            cond0=cond0,
            expNum=j,
        )
        output_fitted = output_fitted.assign(
            dataset=dataset,
            cond1=cond1,
            cond0=cond0,
            expNum=j
        )
        
        this_binned_angles = distribution_binned_average(
            df=rep_raw,
            xname=x_name,
            yname=y_name,
            bins=x_range).assign(
                dataset=dataset,
                cond1=cond1,
                cond0=cond0,
                expNum=j
            ) 

        reg_res = pd.concat([reg_res,output_coef], ignore_index=True)
        model_res = pd.concat([model_res,output_fitted], ignore_index=True)
        binned_angles = pd.concat([binned_angles, this_binned_angles], ignore_index=True)

reg_res = reg_res.assign(
    y_intersect = reg_res['b']**2 * reg_res['k'] + reg_res['c'],
)

#%% plot
binned_angles = binned_angles.assign(
    bin_mids = list((x_range[1:]+x_range[:-1]) /2) * binned_angles.groupby(['cond0', 'cond1','expNum','dataset']).ngroups,
)
binned_angles = binned_angles.sort_values(by=['cond0', 'cond1','expNum','dataset'])
# %
g = sns.relplot(
    data=model_res,
    x='xrange',
    y='yfit',
    color='black',
    hue='cond1',
    col='cond0',
    kind='line',
    height=3,
    errorbar=None,
)
g.map(sns.lineplot, data=binned_angles, x='bin_mids', y=y_name, alpha=0.2,
                    hue='cond1', errorbar='sd'
    )
    # ax.legend([],[], frameon=False)

if y_name == 'spd_peak':
    g.set(
        xlim=[-35, 45],
        ylim=[6, np.percentile(df_toplt[y_name],98)],
        xticks=[-20, 0, 20, 40]
    )   
else:
    g.set(
        xlim=[-30, 45],
        ylim=[np.percentile(df_toplt[y_name],1), np.percentile(df_toplt[y_name],99)],
        xticks=[-20, 0, 20, 40]
    )   
plt.savefig(fig_dir+f"/{y_name}X{x_name} reg.pdf",format='PDF')


# %%
plt_categorical_grid2(
    data=reg_res.sort_values(by='cond1'),
    gridcol='cond0',
    x_name='cond1',
    y_name='k',
    units='expNum',
)
plt.savefig(fig_dir+f"/{y_name}X{x_name} otog coef.pdf",format='PDF')

x = reg_res.loc[reg_res['cond1'] == reg_res.cond1.unique()[0], 'k']
y = reg_res.loc[reg_res['cond1'] == reg_res.cond1.unique()[1], 'k']
print(st.ttest_rel(x, y))

# %%
