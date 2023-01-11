'''
Plot Inter Bout Interval (IBI) posture distribution and standard deviation

This scripts looks for "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" data which includes mean of body angles during IEI

This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, IBI posture of the current experiment will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. if input directory is a folder with subfolders containing dlm data, IBI posture of all the experiments and mean of standard deviation of IBI pitch will be plotted. 
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
NOTE
User may define the number of bouts sampled from each experimental repeat by defining the argument "sample_bout"
Default is off (sample_bout = -1)'''

# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import (set_font_type, day_night_split, round_half_up, setup_vis_parameter, defaultPlotting)

def plot_IBIposture(root, **kwargs):
    """Plot Inter Bout Interval (IBI) posture distribution and standard deviation

    Args:
        root (string): directory
        ---kwargs---
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off
        figure_dir (str): directory to save figures. If not defined, figures will be saved to folder "figures"
    """
    
    print('------\n+ Plotting inter-bout-interval posture')
    # generate figure folder
    folder_name = 'IBI posture'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    root, all_dir, fig_dir, if_sample, SAMPLE_N, if_multiple_repeats = setup_vis_parameter(root, fig_dir, if_sample=False, SAMPLE_N=-1, if_multiple_repeats=False, **kwargs)

    bins = list(range(-90,94,4))

    # %%
    # main function
        
    # %%
    all_angles = pd.DataFrame()
    ang_STD = []
    
    all_dir.sort()
    # go through each condition folders under the root
    for expNum, exp_path in enumerate(all_dir):
        # for each sub-folder, get the path
        df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')
        df = day_night_split(df,'propBoutIEItime')

        # get pitch
        body_angles = df.loc[:,['propBoutIEI_pitch']].assign(expNum = expNum)
        if if_multiple_repeats == True:
            if if_sample == True:
                try:
                    body_angles = body_angles.sample(n=SAMPLE_N)
                except:
                    body_angles = body_angles.sample(n=SAMPLE_N,replace=True)
        all_angles = pd.concat([all_angles, body_angles],axis=0)

    if if_multiple_repeats:
        rep_list = all_angles['expNum'].unique()
        ang_distribution = pd.concat(
            [pd.DataFrame(
                np.histogram(all_angles.loc[all_angles['expNum'] == this_expNum,'propBoutIEI_pitch'].to_numpy().flatten(), bins=bins, density=True)
                ) for this_expNum in rep_list], axis=1
            ).transpose()
        ang_STD = [np.nanstd(all_angles.loc[all_angles['expNum'] == this_expNum,'propBoutIEI_pitch'].to_numpy().flatten()) for this_expNum in rep_list]
        ang_mean = [np.nanmean(all_angles.loc[all_angles['expNum'] == this_expNum,'propBoutIEI_pitch'].to_numpy().flatten()) for this_expNum in rep_list]
        
        ang_STD = pd.DataFrame(data={'STD of posture':ang_STD, 
                                    'expNum':np.arange(0,expNum+1)})
    else:
        ang_distribution = pd.DataFrame(np.histogram(all_angles['propBoutIEI_pitch'].to_numpy().flatten(), bins=bins, density=True)).T
        ang_STD = np.nanstd(all_angles['propBoutIEI_pitch'])
        ang_mean = np.nanmean(all_angles['propBoutIEI_pitch'])

    ang_distribution.columns = ['Probability','Posture (deg)']
    ang_distribution.reset_index(drop=True,inplace=True)

    
    # %%
    # Plot posture distribution and its standard deviation
    # defaultPlotting()
    # set_font_type()

    # plot distribution
    g = sns.lineplot(x='Posture (deg)',
                     y='Probability', 
                     data=ang_distribution, 
                     errorbar='sd', err_style='band')
    g.set_xticks(np.arange(-90,135,45))  # adjust x ticks
    filename = os.path.join(fig_dir, "inter bout interval pitch distribution.pdf")
    plt.savefig(filename,format='PDF')
    
    table_filename = os.path.join(fig_dir,f"STD of IBI pitch.csv")
    all_angle_filename = os.path.join(fig_dir,f"all IBI pitch stats.csv")

    if if_multiple_repeats:
        g = sns.catplot(data = ang_STD,y = 'STD of posture',
                        height=4, aspect=0.8, kind='point',
                        markers='d',sharey=False,
                        )
        g = sns.despine(trim=False)
        filename = os.path.join(fig_dir, "inter-bout interval STD(pitch).pdf")
        plt.savefig(filename,format='PDF')
        
        output_IBIstd = pd.DataFrame(data={
            'IBIpitch_mean_byRepeat': ang_mean,
            'IBIpitch_std_byRepeat': ang_STD['STD of posture'].values
        })
    else:
        output_IBIstd = pd.DataFrame(data={
            'IBIpitch_mean_byRepeat': ang_mean,
            'IBIpitch_std_byRepeat': ang_STD,
        },index=[0])
        
    all_angles.iloc[:,0].describe().to_csv(all_angle_filename)
    output_IBIstd.describe().to_csv(table_filename)

    

if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_IBIposture(root)

# %%
