#%%
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, plot_pointplt)
from plot_functions.get_bout_kinematics import get_bout_kinematics
from statsmodels.stats.multicomp import MultiComparison


def Fig7_bkg_steering_righting(root):
    set_font_type()
    defaultPlotting()
    # %%
    # for day night split
    which_zeitgeber = 'day' # day night all
    SAMPLE_NUM = 0
    print("- Figure 7: ZF strains - Steering & Righting")

    FRAME_RATE = 166
    folder_name = f'Steering and Righting'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass

    # %%
    all_kinetic_cond, kinematics_jackknife, kinematics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinematics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
    all_cond1.sort()
    all_cond2.sort()

    # %% Compare by condition
    toplt = all_kinetic_cond
    cat_cols = ['jackknife_group','condition','expNum','condition0','ztime']
    all_features = [c for c in toplt.columns if c not in cat_cols]
    # print('plot jackknife data')

    for feature_toplt in (all_features):
        print(f"Plotting {feature_toplt}: ")
        plot_pointplt(toplt, feature_toplt, all_cond2)

        filename = os.path.join(fig_dir,f"{feature_toplt}.pdf")
        plt.savefig(filename,format='PDF')
        
        print(f"{feature_toplt}: ")
        multi_comp = MultiComparison(toplt[feature_toplt], toplt['condition'])
        print(multi_comp.tukeyhsd().summary())
        
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig7_bkg_steering_righting(root)