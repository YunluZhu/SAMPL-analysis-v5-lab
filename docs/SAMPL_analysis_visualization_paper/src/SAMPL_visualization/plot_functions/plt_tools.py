import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from decimal import Decimal
import decimal
import os

def round_half_up(var):
    """round half up

    Args:
        var (string or float): value to round to int

    Returns:
        int: rounded int
    """
    res = int(Decimal(var).quantize(Decimal('0'), rounding=decimal.ROUND_HALF_UP))
    return res

def set_font_type():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['savefig.bbox'] = 'tight' 

def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'medium',"ytick.labelsize":'medium', "axes.labelsize":'medium'},style="whitegrid")

def jackknife_list(ori_list):
    matrix = np.tile(ori_list,(len(ori_list),1))
    output = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return output

def jackknife_mean(df):
    output = pd.DataFrame()
    for i in list(df.index):
        output = pd.concat([output, df.loc[df.index != i,:].\
            mean().to_frame().T])
    return output

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    return df_day

def setup_vis_parameter(root, fig_dir, if_sample=False, SAMPLE_N=-1, if_multiple_repeats=False, **kwargs):
    """Prepare parameters for plotting. Make figure directory. Decide whether to sample data from each experimental repeat.

    Args:
        root (str): data directory
        fig_dir (str): directory to save figures
        if_sample (bool, optional): whether to sample from each repeat. Defaults to False.
        SAMPLE_N (int, optional): number of bouts to sample from each repeat. Defaults to -1.
        if_multiple_repeats (bool, optional): whether root dir contain multiple exp repeats. Defaults to False.

    """
    for key, value in kwargs.items():
        if key == 'sample_bout':
            SAMPLE_N = int(value)
        if key == 'figure_dir':
            if value:
                fig_dir = value
            
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        all_dir = all_dir[1:]
        if_multiple_repeats = True
    
    if (SAMPLE_N == -1) & (if_multiple_repeats):
        SAMPLE_N = round_half_up(input("- How many bouts to sample from each repeat? ('0' for no sampling): "))
        
    if SAMPLE_N > 0:
        if_sample = True

    try:
        os.makedirs(fig_dir)
        print(f'- Figure folder created: {fig_dir}')
    except:
        print(f'- Figure directory: {fig_dir}')
        
    return root, all_dir, fig_dir, if_sample, SAMPLE_N, if_multiple_repeats