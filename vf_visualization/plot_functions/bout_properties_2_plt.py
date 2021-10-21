import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import math
import scipy

from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")


def plt_KDE(all_data_cond,hue_order):
## # Posture change - attack angle KDE joint plot

    # # This is a simple kde plot
    # # p = sns.relplot(data=all_data_cond, x='pre_posture_chg',y='atk_ang',col='condition',row='dpf',alpha=0.1,kind='scatter')
    # # p.set(xlim=(-20, 20), ylim=(-20, 25))

    # # This is the joint plot
    # flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())


    # df = all_data_cond

    # plt_condition = hue_order
    # plt_dpf = ['7','7']

    # for i in range(2):
    #     df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
    #     print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
    #     sns.jointplot(df_to_plot['pre_posture_chg'], df_to_plot['atk_ang'], kind="kde", height=5, space=0, xlim=(-12, 12), ylim=(-20, 25))
    # plt.show()

    # # %%
    # # pre_pitch - righting rot KDE joint plot
    # flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())


    # df = all_data_cond

    # plt_condition = hue_order
    # plt_dpf = ['7','7']

    # for i in range(2):
    #     df_to_plot = df.loc[(df['dpf']==plt_dpf[i]) & (df['condition']==plt_condition[i]),:]
    #     print(f'* {plt_dpf[i]} dpf | {plt_condition[i]}')
    #     sns.jointplot(df_to_plot['pre_pitch'], df_to_plot['decel_rot'], kind="kde", height=5, space=0, xlim=(-12, 12), ylim=(-20, 25))
    # # plt.show()

    # # pitch - pre_pitch
    # flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())
    # df = all_data_cond
    # plt_condition = hue_order
    
    flatui = ["#D0D0D0"] * (all_data_cond.groupby('condition').size().max())
    df = all_data_cond
    plt_condition = hue_order

    # for i, cur_cond in enumerate(plt_condition):
    #     df_to_plot = df.loc[df['condition']==cur_cond,:]
    #     print(f'{plt_condition[i]}')
    #     sns.jointplot(df_to_plot['pitch'], df_to_plot['pre_pitch'], kind="kde", height=5, space=0)

    # pre_pitch - pre change

    for i, cur_cond in enumerate(plt_condition):
        df_to_plot = df.loc[df['condition']==cur_cond,:]
        print(f'{plt_condition[i]}')
        sns.jointplot(df_to_plot['pre_pitch'], df_to_plot['pre_posture_chg'], kind="kde", height=5, space=0,
                          xlim=(-40, 10), ylim=(-10, 10))
        
    # # # pre_pitch - steering
    # for i, cur_cond in enumerate(plt_condition):
    #     df_to_plot = df.loc[df['condition']==cur_cond,:]
    #     print(f'{plt_condition[i]}')
    #     sns.jointplot(df_to_plot['pre_pitch'], df_to_plot['accel_rot'], kind="kde", height=5, space=0)

    # # # pre_pitch - righting
    # for i, cur_cond in enumerate(plt_condition):
    #     df_to_plot = df.loc[df['condition']==cur_cond,:]
    #     print(f'{plt_condition[i]}')
    #     sns.jointplot(df_to_plot['pre_pitch'], df_to_plot['decel_rot'], kind="kde", height=5, space=0)

    # # decel_ang - speed
    for i, cur_cond in enumerate(plt_condition):
        df_to_plot = df.loc[df['condition']==cur_cond,:]
        print(f'{plt_condition[i]}')
        sns.jointplot(df_to_plot['speed'], df_to_plot['accel_rot'], kind="kde", height=5, space=0)

    # # accel_ang - speed
    for i, cur_cond in enumerate(plt_condition):
        df_to_plot = df.loc[df['condition']==cur_cond,:]
        print(f'{plt_condition[i]}')
        sns.jointplot(df_to_plot['speed'], df_to_plot['decel_rot'], kind="kde", height=5, space=0)


def plt_meanData(mean_data_cond,hue_order):
    flatui = ["#D0D0D0"] * (mean_data_cond.groupby('condition').size().max())

    # %%
    # plot mean attack angles, mean max speed, mean posture change (Figure 1—figure supplement 3)

    multi_comp = MultiComparison(mean_data_cond['atkAng'], mean_data_cond['dpf']+mean_data_cond['condition'])
    print('* attack angles')
    print(multi_comp.tukeyhsd().summary())
    multi_comp = MultiComparison(mean_data_cond['maxSpd'], mean_data_cond['dpf']+mean_data_cond['condition'])
    print('* max Speed')
    print(multi_comp.tukeyhsd().summary())
    multi_comp = MultiComparison(mean_data_cond['meanRot'], mean_data_cond['dpf']+mean_data_cond['condition'])
    print('* mean rotation')
    print(multi_comp.tukeyhsd().summary())

    defaultPlotting()
    fig, axs = plt.subplots(3)
    fig.set_figheight(15)
    fig.set_figwidth(4)

    for i, ax in enumerate(axs):
        g = sns.pointplot(x='condition',y=mean_data_cond.iloc[:,i], hue='date',data=mean_data_cond,
                    palette=sns.color_palette(flatui), scale=0.5,
                    order= hue_order,
                    ax=ax)
        g = sns.pointplot(x='condition', y=mean_data_cond.iloc[:,i],hue='condition',data=mean_data_cond, 
                    linewidth=0,
                    alpha=0.9,
                    order=hue_order,
                    ci=None,
                    markers='d',
                    ax=ax
                    )
        # p-value calculation
        ttest_res, ttest_p = ttest_rel(mean_data_cond.loc[mean_data_cond['condition']=='Sibs',mean_data_cond.columns[i]],
                                    mean_data_cond.loc[mean_data_cond['condition']=='Tau',mean_data_cond.columns[i]])
        print(f'{mean_data_cond.columns[i]} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')

        g.legend_.remove()

    plt.show()
