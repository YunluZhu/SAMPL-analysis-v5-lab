'''
Multiple linear regression to extract classifiers that review changes in depth change during swimming (spd > 5mm/s)
Retionale: navigation in depth is central to larvae locomotion while requires high energy consumption
Steps:
1. determine features correlates with ydispl_swim ~  pitch_peak + yvel_peak + traj_peak + rot_l_decel + rot_full_accel
2. run multiple linear regression
3. use residules as classifier. Positive: empirical > fitted
4. plot for all bouts with different categories (day/night, ...)
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
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid
from plot_functions.get_bout_kinetics import get_bout_kinetics
from statsmodels.stats.multicomp import MultiComparison
from sklearn import metrics
import scipy.stats as st
from tqdm import tqdm
import matplotlib as mpl

from pymer4.models import Lm
from pymer4.models import Lmer

set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'astro'
which_ztime = 'all'
spd_bins = np.arange(5,30,5)

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} associate features'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_ztime,speed_bins=spd_bins)
all_feature_cond = all_feature_cond.assign(
    exp_condition = all_feature_cond['cond0']+'&'+all_feature_cond['cond1']
    )
all_feature_cond = all_feature_cond.assign(
    expNum_cond = all_feature_cond['expNum'].astype('str')+'&'+all_feature_cond['exp_condition']
    )
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

all_feature_cond = all_feature_cond.assign(
    yvel_initial_phased = np.sin(np.deg2rad(all_feature_cond['traj_initial_phase'])) * all_feature_cond['spd_initial_phase'],
    vel_peak = all_feature_cond['traj_peak']/all_feature_cond['traj_peak'].abs() * all_feature_cond['spd_peak'],
    xvel_peak = np.cos(np.deg2rad(all_feature_cond['traj_peak'])) * all_feature_cond['spd_peak'],
    yvel_peak = np.sin(np.deg2rad(all_feature_cond['traj_peak'])) * all_feature_cond['spd_peak'],

)

pitch_bins = np.arange(-50,42,20)
spd_bins = np.arange(5,30,8)

all_feature_cond = all_feature_cond.assign(
    initial_bins = pd.cut(all_feature_cond['pitch_initial'],pitch_bins),
    direction = pd.cut(all_feature_cond['pitch_initial'], 
                       bins=[-90,10,91],labels=['nose_dn','nose_up']),
    
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins),
    vel_direction = pd.cut(all_feature_cond['pitch_initial'], 
                       bins=[-np.inf,0,np.inf],labels=['vel_dn','vel_up']),
    
    righting_bins = pd.cut(all_feature_cond['rot_l_decel'], 
                       bins=[-90,0,91],labels=['right_dn','right_up']),
    steering_bins = pd.cut(all_feature_cond['rot_full_accel'], 
                       bins=[-90,0,91],labels=['steer_dn','steer_up']),
)

# %%

input_df = all_feature_cond[[
                             'ydispl_swim',
                             'depth_chg',
                            # 'xdispl_swim',
                             'pitch_peak',
                            #  'vel_peak',
                            #  'xvel_peak',
                             'yvel_peak',
                             'traj_peak',
                            #  'x_chg',
                             'rot_l_decel',
                             'rot_full_accel',
                            #  'bout_trajectory_Pre2Post',
                             'angvel_chg',
                            #  'angvel_prep_phase',
                            #  'angvel_post_phase',
                            #  'angvel_initial_phase',
                            #  'spd_initial_phase',
                            #  'rot_total',
                            #  'rot_l_accel',
                            #  'expNum',
                            #  'exp_condition',
                            #  'ztime',
                            #  'direction'
                             ]]

# %%
sns.pairplot(data = input_df,kind='hist')
filename = os.path.join(fig_dir,f"features pair plot.pdf")
plt.savefig(filename,format='PDF')


# %%
output_coef = pd.DataFrame() 
output_data = pd.DataFrame()

df_all = all_feature_cond
df_day = all_feature_cond.loc[all_feature_cond['ztime']=='day']
df_night = all_feature_cond.loc[all_feature_cond['ztime']=='night']

# df = all_feature_cond
df = df_all
# feature_list = ['ztime','direction','expNum'] # for one experimental condition 
feature_list = ['exp_condition' ,'ztime'] # categorical 

for sel_feature in feature_list:
    all_conditions = df[sel_feature].unique()
    for condition in all_conditions:
        if condition == condition:
            this_df = df.loc[df[sel_feature] == condition]
            # model = Lm("depth_chg ~  pitch_peak + vel_peak + traj_peak + x_chg + rot_l_decel + rot_full_accel", data=this_df)
            model = Lm("ydispl_swim ~  pitch_peak + yvel_peak + traj_peak + rot_l_decel + rot_full_accel", data=this_df)
            model.fit()
            this_output = model.coefs.reset_index().assign(feature_cond = condition, sel_feature = sel_feature)
            output_coef = pd.concat([output_coef,this_output],ignore_index=True)
            
            this_data = model.data.reset_index().assign(feature_cond = condition, sel_feature = sel_feature)
            output_data = pd.concat([output_data,this_data],ignore_index=True)


g = sns.catplot(
    data = output_coef,
    x = 'feature_cond',
    y = 'Estimate',
    row = 'index',
    col = 'sel_feature',
    height = 3,
    aspect = 0.8,
    sharex = False,
    sharey = 'row',
)
g.set_titles("{row_name}|{col_name}")

filename = os.path.join(fig_dir,f"multiLinearRegression res.pdf")
plt.savefig(filename,format='PDF')


# # %%
# ############### model classifier
# def intercept_calc(data, target_feature, feature_val, estimator_val):
#     calculator = [0] * len(data)
#     for (coef, feature) in zip(estimator_val, feature_val):
#         if feature != 'Intercept':
#             calculator += data[feature].values * coef
#     return data.assign(classifier = data[target_feature].values - calculator)

# # %% take mean of by_expNum
# by_exp_coef = output_coef.loc[output_coef['sel_feature']=='expNum']
# by_exp_coef_mean = by_exp_coef.groupby(['index']).median()
# classifier_res = intercept_calc(all_feature_cond,'depth_chg', by_exp_coef_mean.index, by_exp_coef_mean['Estimate'].values)

# %% take the regression res using all data

model = Lm("ydispl_swim ~  pitch_peak + vel_peak + yvel_peak + traj_peak + rot_l_decel + rot_full_accel", data=df_day)
model.fit()
output_coef = model.coefs
output_data = model.data

# %%
# classifier_res = intercept_calc(all_feature_cond,'depth_chg', output_coef.index, output_coef['Estimate'].values)
classifier_res = all_feature_cond.assign(
    classifier = output_data['residuals']
)
# %%
feature_of_interest = 'righting_bins'
hue_feature = 'exp_condition'
g = sns.catplot(data = classifier_res,
                kind = 'point',
                hue = hue_feature,
                y = 'classifier',
                x = feature_of_interest,
                # col = 'righting_bins',
                height = 3,
                # aspect = 1.2,
                join=False)

filename = os.path.join(fig_dir,f"classifier by_{feature_of_interest} X {hue_feature}.pdf")
plt.savefig(filename,format='PDF')

feature_of_interest = 'initial_bins'
hue_feature = 'exp_condition'
g = sns.catplot(data = classifier_res,
                kind = 'point',
                hue = hue_feature,
                y = 'classifier',
                x = feature_of_interest,
                # col = 'initial_bins',
                height = 3,
                # aspect = 1.2,
                join=False)

filename = os.path.join(fig_dir,f"classifier by_{feature_of_interest} X {hue_feature}.pdf")
plt.savefig(filename,format='PDF')

feature_of_interest = 'direction'
hue_feature = 'exp_condition'
g = sns.catplot(data = classifier_res,
                kind = 'point',
                hue = hue_feature,
                y = 'classifier',
                x = feature_of_interest,
                # col = 'initial_bins',
                height = 3,
                # aspect = 1.2,
                join=False)

filename = os.path.join(fig_dir,f"classifier by_{feature_of_interest} X {hue_feature}.pdf")
plt.savefig(filename,format='PDF')
# %%
feature_of_interest = 'SteerRight'
hue_feature = 'exp_condition'
g = sns.catplot(data = classifier_res.assign(SteerRight=classifier_res['steering_bins'].astype('str')+'|'+classifier_res['righting_bins'].astype('str')),
                kind = 'point',
                hue = hue_feature,
                y = 'classifier',
                x = feature_of_interest,
                # col = 'initial_bins',
                height = 3,
                # aspect = 1.2,
                join=False)

filename = os.path.join(fig_dir,f"classifier by_{feature_of_interest} X {hue_feature}.pdf")
plt.savefig(filename,format='PDF')
# if two conditions
conditions_forttest = classifier_res[feature_of_interest].unique()
st.ttest_ind(classifier_res.loc[classifier_res[feature_of_interest]==conditions_forttest[0],'classifier'],
             classifier_res.loc[classifier_res[feature_of_interest]==conditions_forttest[1],'classifier'])

# %%
# %%
# # model = Lm("depth_chg ~ rot_full_accel + rot_l_decel + pitch_peak + spd_peak", data=df)
# model = Lm("depth_chg ~  pitch_peak + expNum", data=df_day)
# print(model.fit())
# sns.regplot(x="fits", y="depth_chg", data=model.data, fit_reg=True)

# %%
# model = Lmer("depth_chg ~  pitch_peak + spd_peak + (pitch_peak + spd_peak | expNum)", data=df)
# print(model.fit())
# sns.regplot(x="fits", y="depth_chg", data=model.data, fit_reg=True)
# model.plot_summary()

# # %% interesting, look at those empirical data that are greater than fits
# sel_data = model.data.loc[(model.data['depth_chg']>model.data['fits']) & (model.data['depth_chg']>1)]

# df = df.assign(
#     sel_data = False
# )
# df.loc[sel_data.index,'sel_data'] = True

# # %%
# model = Lmer("depth_chg ~  pitch_peak + spd_peak + spd_initial_phase +  (pitch_peak + spd_peak + spd_initial_phase | expNum)", data=df)
# print(model.fit())
# sns.regplot(x="fits", y="depth_chg", data=model.data, fit_reg=True)
# model.plot_summary()
 # %%
# x_feature = 'yvel_initial_phased'
# upper = np.percentile(df[x_feature], 99)
# lower = np.percentile(df[x_feature], 0.5)
# g = sns.kdeplot(
#    data=df, x=x_feature, hue="ztime",
#    common_norm=False,
#    clip = (lower,upper)
# )
# plt.axvline(x=0,color='black')
# # %%
# x_feature = 'angvel_initial_phase'
# upper = np.percentile(df[x_feature], 99)
# lower = np.percentile(df[x_feature], 0.5)
# g = sns.kdeplot(
#    data=df, x=x_feature, hue="ztime",
#    common_norm=False,
#    clip = (lower,upper)
# )

# %%

# %%
