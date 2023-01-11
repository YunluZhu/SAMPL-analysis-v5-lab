

# %%
import os, glob
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_tools import (set_font_type)
from plot_functions.get_data_dir import (get_figure_dir)
from scipy.signal import savgol_filter


from tqdm import tqdm

def Fig2_fly_worm_epoch(master_root):
    # %% features for plotting
    all_features = {
        'ang_calc':'angle (deg)',
        # 'absy':'y position (mm)'
        # 'deltaT', 
        'x':'x',
        'y':'z (mm)',
        'headx':'head x (mm)',
        'heady':'head y (mm)',
        # 'centeredAng':'centered angle (deg)',
        # 'xvel', 
        'yvel': 'yvel (mm*s-1)',
        'dist':'distance (mm)',
        # 'displ':'displacement (mm)',
        # 'angVel':'ang vel (deg*s-1)',
        # 'angVelSmoothed', 
        # 'angAccel':'ang accel (deg*s-2)',
        'speed':'speed (mm*s-1)',
        # 'velocity':'velocity (mm*s-1)'
    }
    # %%
    # generate figure folder
    folder_name = f'Single epoch speed and pitch - fly worm'
    folder_dir2 = get_figure_dir('Fig_2')
    fig_dir2 = os.path.join(folder_dir2, folder_name)

    try:
        os.makedirs(fig_dir2)
    except:
        pass

    # %%
    for organism in ['Drosophila', 'C elegans']:
        root = os.path.join(master_root,organism)
        # %%
        # Read dlm
        filenames = glob.glob(os.path.join(root,"*.dlm"))
        col_names = ['time','fishNum','ang_adj','rawx','rawy','headx','heady','rawAng','epochNum','length']

        # %%
        filename = filenames[0]
        try:
            # raw = pd.read_csv(filename, sep="\t",names = col_names) # load .dlm
            raw = pd.read_csv(filename, sep="\t",header=None)
        except FileNotFoundError:
            print(f"No .dlm file found in the directory entered")
        else:
            pass

        if raw.shape[1] > 1: # if data CRLF is correct
            raw.columns = col_names
        else: # if data only comes in one column, legacy V2 program debug code
            raw_reshaped = pd.DataFrame(np.reshape(raw.to_numpy(),(-1,10)), columns = ['time','fishNum','ang','absx','absy','absHeadx','absHeady','epochNum','col7','fishLen']) # reshape 1d array to 2d
            # assuming timestamp is not saved correctly
            raw_reshaped['time'] = np.arange(0,1/160*raw_reshaped.shape[0],1/160)
            # edit fish number
            raw_reshaped['fishNum'] = raw_reshaped['fishNum']-1
            raw = raw_reshaped

        # if from gen2 program, fish num == 1 for 1 fish detected, change that to 0 
        if raw['fishNum'].min() > 0:
            raw['fishNum'] = raw['fishNum']-1
            
        # Clear original time data stored in the first row
        raw.loc[0,'time'] = 0
        # data error results in NA values in epochNum, exclude rows with NA
        raw.dropna(inplace=True)
        # rows with epochNum == NA may have non-numeric data recorded. In this case, change column types to float for calculation. not necessary for most .dlm.
        # raw[['animalNum','ang','x','y','headx','heady','full_ang','epochNum','length']] = raw[['fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']].astype('float64',copy=False)

        # %%
        #  find the epoch
        SCALE = 60
        FRAME_RATE = 166
        if organism == 'Drosophila':
            which_epoch = 36
        elif organism == 'C elegans':
            which_epoch = 3622

        raw.groupby('epochNum').size()
        epoch_sel = raw.loc[raw['epochNum']==which_epoch]
        # g = sns.scatterplot(data = epoch_sel, x = 'absx', y = 'absy')
        # g.set(
        #     xlim=[0,1200],
        #     ylim=[0,1200]
        # )
        epoch_sel = epoch_sel.assign(
            x = savgol_filter(epoch_sel['rawx'],5,3),
            y = savgol_filter(epoch_sel['rawy'],5,3),
        )
        epoch_sel = epoch_sel.assign(
            dist = np.linalg.norm(epoch_sel[['x','y']].diff(), axis=1),
            deltaT = raw['time'].diff()
        ).reset_index(drop=True)
        epoch_sel.loc[0,'deltaT'] = 1/FRAME_RATE

        epoch_sel.loc[:,['y','heady','yvel']] = epoch_sel[['y','heady']] * -1 / SCALE
        epoch_sel.loc[:,['x','headx','dist','length']] = epoch_sel[['x','headx','dist','length']] / SCALE

        epoch_sel = epoch_sel.assign(
            speed = epoch_sel['dist']/epoch_sel['deltaT'],
            ang_calc = savgol_filter(epoch_sel['rawAng'],5,3),
            yvel = epoch_sel['y'].diff()/epoch_sel['deltaT'],
        )



        # %%

        data_toplt = epoch_sel.loc[:,all_features.keys()]
        data_toplt = data_toplt.rename(columns=all_features)

        data_toplt = data_toplt.assign(
            time_s = np.cumsum(epoch_sel['deltaT'])
        )
        if organism == 'Drosophila':
            data_toplt['angle (deg)'] = data_toplt.loc[:,'angle (deg)']-90 # make north = 0 deg
            start_i = 2
            data_toplt = data_toplt.iloc[start_i:start_i+400]
            set_font_type()
            data_toplt = data_toplt.rename(columns = {'angle (deg)':'heading (deg)'})
            features_toplot = ['speed (mm*s-1)', 'heading (deg)']
            for feature_toplt in features_toplot:
                p = sns.relplot(
                    data = data_toplt, x = 'time_s', y = feature_toplt,
                    kind = 'line',aspect=2.5, height=2
                    )
                plt.savefig(os.path.join(fig_dir2, f"{organism}_{feature_toplt}_raw.pdf"),format='PDF')

        elif organism == 'C elegans':
            data_toplt['angle (deg)'] = 90 - data_toplt.loc[:,'angle (deg)'] # make north = 0 deg
            start_i = 2
            data_toplt = data_toplt.iloc[start_i:start_i+5000]
            set_font_type()
            data_toplt = data_toplt.rename(columns = {'angle (deg)':'approximate posture (deg)'})
            features_toplot = [
                            'approximate posture (deg)',
                        'z (mm)']
        for feature_toplt in features_toplot:
            p = sns.relplot(
                data = data_toplt, x = 'time_s', y = feature_toplt,
                kind = 'line',aspect=2.5, height=2
                )
            plt.savefig(os.path.join(fig_dir2, f"{organism}_{feature_toplt}_raw.pdf"),format='PDF')


# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'fly worm data')
    Fig2_fly_worm_epoch(root)