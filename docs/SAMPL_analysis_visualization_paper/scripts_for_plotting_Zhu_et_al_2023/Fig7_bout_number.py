#%%
# import sys
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_bout_features import get_bout_features
import os

def Fig7_bout_number(root):
    set_font_type()
    defaultPlotting()
    # for day night split
    which_zeitgeber = 'day' # day night all
    FRAME_RATE = 166
    # %%
    # all_kinetic_cond, kinematics_jackknife, kinematics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinematics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
    IBI_angles, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber )#, max_angvel_time = max_angvel_time)

    # %%
    print("Total bout number per strain:")
    print(all_feature_cond.groupby('condition').size())
    print("Mean bout number by run (experimental repeat):")
    print(f"{all_feature_cond.groupby(['condition','expNum']).size().reset_index(drop=True).mean():.2f}±{all_feature_cond.groupby(['condition','expNum']).size().reset_index(drop=True).std():.2f}")
    print("\nIBI number:")
    print(IBI_angles.groupby('condition').size())
# %%
if __name__ == "__main__":
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')  
    Fig7_bout_number(root)