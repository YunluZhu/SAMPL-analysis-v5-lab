#%%
# import sys
import os
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type)
import matplotlib as mpl

def Fig2_parameter_distribution_prob(root):
    set_font_type()
    mpl.rc('figure', max_open_warning = 0)

    which_ztime = 'day'
    FRAME_RATE = 166

    folder_name = f'Parameter distribution'
    folder_dir2 = get_figure_dir('Fig_2')
    fig_dir2 = os.path.join(folder_dir2, folder_name)

    try:
        os.makedirs(fig_dir2)
    except:
        pass
    # %% get features
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
    all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    # %% tidy data

    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    all_feature_UD = all_feature_cond

    # %%
    # Plot parameter distribution
    print("Figure 2: Distribution of parameters")
    toplt = all_feature_UD
    feature_to_plt = ['pitch_initial','pitch_peak','pitch_post_bout','spd_peak','rot_total','traj_peak',
                    'bout_displ','bout_traj']

    for feature in feature_to_plt:
        # let's add unit
        if 'spd' in feature:
            xlabel = feature + " (mm*s^-1)"
        elif 'dis' in feature:
            xlabel = feature + " (mm)"
        else:
            xlabel = feature + " (deg)"
        
        upper = np.percentile(toplt[feature], 99.5)
        lower = np.percentile(toplt[feature], 0.5)
        
        if 'spd' in feature:
            lower = toplt[feature].min()

        plt.figure(figsize=(3,2))
        g = sns.histplot(data=toplt, x=feature, 
                            bins = 30, 
                            element="poly",
                            #  kde=True, 
                            stat="probability",
                            pthresh=0.05,
                            binrange=(lower,upper),
                            color='grey'
                            )
        g.set_xlabel(xlabel)
        sns.despine()
        plt.savefig(fig_dir2+f"/{feature} distribution.pdf",format='PDF')
        print(f"{feature}: {toplt[feature].mean():.3f} ± {toplt[feature].std():.3f}")

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    
    Fig2_parameter_distribution_prob(root)