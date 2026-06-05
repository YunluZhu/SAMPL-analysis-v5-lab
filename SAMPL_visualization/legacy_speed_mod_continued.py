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
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split, get_2sd,jackknife_list)
from plot_functions.get_bout_kinetics import get_bout_kinetics
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel
from lmfit import Model

set_font_type()

def linear_func(x, a, b, **kwargs):
    # parabola function
    return a * x + b


def linear_reg(df, yname, xname, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    df = df.sort_values(by=xname)
    df = df[[xname, yname]].dropna()
    mod  = Model(linear_func)
    res = mod.fit(df[yname],x=df[xname],a=0.5, b=-1)
    parameters = res.params.valuesdict()
    pars = parameters.values()
    
    y = [linear_func(x_val, *pars) for x_val in X_RANGE_to_fit]
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    
    y_est = []
    for x in df[xname].values:
        y_est.append(linear_func(x,*pars, log_it=False))
    
    r2 = r2_score(df[yname].values, y_est)
    _ = None
    return pd.DataFrame(data=parameters, index=[0]), output_fitted, _, r2

def distribution_binned_median(df, xname, yname, x_range):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by=xname)
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
    grp = df.groupby(bins)
    df_out = grp[[xname,yname]].median()
    return df_out

# def distribution_binned_zscore(df, xname, yname, x_range):
#     '''
#     bins raw pitch data using fixed bin width. Return binned z of pitch and bout frequency.
#     '''
#     df = df.sort_values(by=xname)
#     # df = df.assign(bout_freq = 1/df['propBoutIEI'])
#     bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
#     grp = df.groupby(bins)
#     df_out = grp[[xname,yname]].median()
#     return df_out

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


# def distribution_binned_FilterByPercentile_mean(df, xname, yname, x_range):
#     '''
#     bins raw pitch data using fixed bin width. Return binned z of pitch and bout frequency.
#     '''
#     df = df.sort_values(by=xname)
#     # df = df.assign(bout_freq = 1/df['propBoutIEI'])
#     bins = pd.cut(df[xname], list(np.arange(x_range.min(),x_range.max(),(x_range.max()-x_range.min())/16)))
#     grp = df.groupby(bins)
#     df_out = grp[[xname,yname]].apply(lambda x: thresh_filter_average(x, yname, yname)).to_frame()
#     df_out.columns = [yname]
#     df_out[xname] = grp[[xname]].median()
#     return df_out

# %%
data_list = ['long21'] # all or specific data
which_zeitgeber = 'day'
if_jackknife = False
if_narrow_time_window = True

data_name = ''.join(data_list)

folder_name = f'spdMod_continued'
folder_dir = get_figure_dir(data_name)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass



df_features_combined = pd.DataFrame()

for pick_data in data_list:
    root, FRAME_RATE = get_data_dir(pick_data)
    all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond['dataset'] = pick_data
    df_features_combined = pd.concat([df_features_combined,all_feature_cond], ignore_index=True)

# df_features_combined = df_features_combined.assign(
#     cond1 = df_features_combined['cond1'].map({'otog': "cond", 
#                                                        'hets':'ctrl',
#                                                        '1ctrl':'ctrl', 
#                                                        '2cond': 'cond', 
#                                                        'lesion':'cond',
#                                                        'ctrl': 'ctrl',
#                                                        'lesionD1':'ctrl', 
#                                                        'lesionD2':'cond', 
#                                                        'finless': 'cond',
#                                                        '1sibs':'ctrl',
#                                                        '2tau':'cond',
#                                                        'ld':'ld',
#                                                        'dd':'dd',
#                                                        })
# )


# %%
df_features_combined = day_night_split(df_features_combined, 'bout_time', narrow_bin=if_narrow_time_window, ztime=which_zeitgeber)
df_features_combined = df_features_combined.assign(
    pitch_peak_abs = df_features_combined['pitch_peak'].abs(),
    yspd_peak = df_features_combined['spd_peak']*np.sin(df_features_combined['bout_trajectory_Pre2Post']* np.pi / 180),
    xspd_peak = df_features_combined['spd_peak']*np.cos(df_features_combined['bout_trajectory_Pre2Post']* np.pi / 180),
    lift_ratio = df_features_combined['lift_distance_fullBout']/df_features_combined['depth_chg_fullBout'],
    propul_distance_fullBout =  df_features_combined['depth_chg_fullBout'] - df_features_combined['lift_distance_fullBout'],
    
    # yspd_peak = df_features_combined['spd_peak']*np.sin(df_features_combined['traj_peak']* np.pi / 180),
    # xspd_peak = df_features_combined['spd_peak']*np.cos(df_features_combined['traj_peak']* np.pi / 180),
    # lift_ratio = df_features_combined['lift_distance_fullBout']/df_features_combined['depth_chg_fullBout'],
    # propul_distance_fullBout =  df_features_combined['depth_chg_fullBout'] - df_features_combined['lift_distance_fullBout'],
    # yposture_spd_peak = df_features_combined['xspd_peak']*np.tan(df_features_combined['pitch_peak']* np.pi / 180),

)
df_features_combined = df_features_combined.assign(
    yposture_spd_peak = df_features_combined['xspd_peak']*np.tan(df_features_combined['pitch_peak']* np.pi / 180),
    lift_index = (df_features_combined['lift_distance_fullBout']-df_features_combined['propul_distance_fullBout'])/df_features_combined['depth_chg_fullBout'],
)
df_features_combined = df_features_combined.assign(
    lift_spd = df_features_combined['yspd_peak'] - df_features_combined['yposture_spd_peak']
    # lift_spd_index = (df_features_combined['lift_spd']-df_features_combined['yposture_spd_peak'])/df_features_combined['yspd_peak'],
)

# df_features_combined = df_features_combined.assign(
#     atk_ang_bins = pd.cut(df_features_combined.atk_ang, bins=[-50,-20,-2,3,20,50]),
#     spd_peak_adj = df_features_combined['spd_peak']
# )
# df_features_combined.loc[df_features_combined.dataset=='otog', 'spd_peak_adj'] = df_features_combined.loc[df_features_combined.dataset=='otog', 'spd_peak_phased'] 


#%% test
toplt = df_features_combined#.loc[df_features_combined['lift_spd_index'].abs()<20]

xname='pitch_peak'
yname = None
yname='spd_peak'

xname='xspd_peak'
yname='yspd_peak'

if yname:
    g = sns.displot(data=toplt,#.groupby('cond1').sample(n=10000).reset_index(),
                    y=yname,
                    x=xname,
                    col='cond0', 
                    row='cond1',
                    hue='cond1',
                    height=3,
                    pthresh=0.02,
                    common_norm=False,
                    )
    g.add_legend()
    g.set(
        ylim=(np.percentile(toplt[yname],.2),np.percentile(toplt[yname],99.8)),
        xlim=(np.percentile(toplt[xname],0.2),np.percentile(toplt[xname],99.8)))
    plt.savefig(fig_dir+f"/dist {yname}X{xname}_.pdf",format='PDF')
    
    p = sns.displot(data=toplt,#.groupby('cond1').sample(n=10000).reset_index(),
                    y=yname,
                    x=xname,
                    col='cond0', 
                    hue='cond1',
                    height=3,
                    pthresh=0.02,
                    level=4,
                    common_norm=False,
                    kind='kde'
                    )
    p.add_legend()
    p.set(
        ylim=(np.percentile(toplt[yname],.2),np.percentile(toplt[yname],99.8)),
        xlim=(np.percentile(toplt[xname],0.2),np.percentile(toplt[xname],99.8)))
    plt.savefig(fig_dir+f"/kde {yname}X{xname}_.pdf",format='PDF')

else:
    g = sns.displot(data=toplt.groupby('cond1').sample(n=10000).reset_index(),
                x=xname,
                col='cond0', 
                hue='cond1',
                height=3,
                common_norm=False,
                kind='kde'
                )
    g.add_legend()
    g.set(
        xlim=(np.percentile(toplt[xname],0.2),np.percentile(toplt[xname],99.8)))


#%%
# %%  fit with parabola, use lmfit module
data = df_features_combined
# data = df_features_combined.query('pitch_peak > 0')
# data = data.query('pitch_peak < 40')

feature_pairs = [
    # ('pitch_peak', 'spd_peak'),
    # ('pitch_peak', 'lift_spd'),
    # ('pitch_peak', 'yposture_spd_peak'),
    # ('pitch_peak', 'xspd_peak'),
    # ('pitch_peak', 'yspd_peak'),
    ('pitch_peak', 'depth_chg_fullBout'),
    ('pitch_peak', 'additional_depth_chg'),
]

for (x_name, y_name) in feature_pairs:

    reg_res = pd.DataFrame()
    model_res = pd.DataFrame()
    binned_angles = pd.DataFrame()
    for (dataset, cond1), this_df in data.groupby(['dataset','cond1']):  
        x_range_min = np.percentile(this_df[x_name],1)
        x_range_max = np.percentile(this_df[x_name],99)
        this_df = this_df.loc[(this_df[x_name]>=x_range_min) & (this_df[x_name]<=x_range_max)]
        x_range = np.arange(x_range_min, x_range_max, 0.5)

        if if_jackknife:
            jackknife_exp_matrix = jackknife_list(this_df.expNum.unique())
        else:
            jackknife_exp_matrix = [[item] for item in this_df.expNum.unique()]
            
        for j, exp_group in enumerate(jackknife_exp_matrix):
            rep_group = this_df.loc[this_df['expNum'].isin(exp_group),:]
            output_coef, output_fitted, p_sigma, r2 = linear_reg(
                df=rep_group,
                xname=x_name,
                yname=y_name,
                X_RANGE_to_fit=x_range
            )
            output_coef.columns=['k', 'b']
            output_fitted.columns=['yfit','xrange']
            output_coef = output_coef.assign(
                # k_sigma=p_sigma[0],
                # b_sigma=p_sigma[1],
                # c_sigma=p_sigma[2],
                r_square=r2,
                dataset=dataset,
                cond1=cond1,
                expNum=j
            )
            output_fitted = output_fitted.assign(
                dataset=dataset,
                cond1=cond1,
                expNum=j
            )

            this_binned_angles = distribution_binned_median(
                df=rep_group,
                xname=x_name,
                yname=y_name,
                x_range=x_range).assign(
                    dataset=dataset,
                    cond1=cond1,
                    expNum=j,
                )  

            reg_res = pd.concat([reg_res,output_coef], ignore_index=True)
            model_res = pd.concat([model_res,output_fitted], ignore_index=True)
            binned_angles = pd.concat([binned_angles, this_binned_angles], ignore_index=True)


    # % plot
    x_name = x_name
    y_name = y_name

    g = sns.relplot(
        data=model_res,
        x='xrange',
        y='yfit',
        color='black',
        row='dataset',
        col='cond1',
        kind='line',
        height=3,
        # aspect=1.2
    )
    for (row_val, col_val), ax in g.axes_dict.items():
        thiscond_data = data.query("dataset==@row_val & cond1==@col_val")
        sns.histplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
                        cbar=True, pthresh=.05,
                        color='black'
        )
        # sns.scatterplot(data=thiscond_data.sample(10000), x=x_name, y=y_name, ax=ax, alpha=0.01,
        #                 hue='cond1',
        #                 # hue_order=['ctrl','cond'],
        # )
        thiscond_data = binned_angles.query("dataset==@row_val & cond1==@col_val")
        sns.lineplot(data=thiscond_data, x=x_name, y=y_name, ax=ax, 
                        color='white', 
                        units='expNum', estimator=None
        )
        # ax.legend()
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


    g.set(
        xlim=[np.percentile(data[x_name],1), np.percentile(data[x_name],99)],
        ylim=[np.percentile(data[y_name],1), np.percentile(data[y_name],99)],
    )    

    plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}.pdf",format='PDF')

    # % coef
    for i, coef_col_name in enumerate(reg_res.columns[:3]):
        p = sns.catplot(
            data = reg_res, y=coef_col_name,x='cond1',
            col='dataset', 
            kind='point',join=False,errorbar='sd',
            hue='cond1', dodge=False,sharey=True,height=3, aspect=0.6
        )
        p.map(sns.lineplot,'cond1',coef_col_name,
            estimator=None,
            units='expNum',
            color='grey',
            alpha=0.2,
            data=reg_res)
        plt.savefig(fig_dir+f"/ind_bouts_{y_name}-{x_name}_{coef_col_name}.pdf",format='PDF')


# %%

