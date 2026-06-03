#%%

import sys
import os
import glob
import pandas as pd
import numpy as np
# from scipy.signal import savgol_filter

#%%
def read_dlm(i, filename):
    """Read .dlm files into a DataFrame

    Args:
        i (int): index of the file in the folder
        filename (string): directory of the .dlm file

    Returns:
        DataFrame: 
    """
    # read_dlm takes file index: i, and the file name end with .dlm
    # col 1-7
    col_names = ['time','col2','col3','col4','col5','col6','col7']
    try:
        # raw = pd.read_csv(filename, sep="\t",names = col_names) # load .dlm
        raw = pd.read_csv(filename, sep="\t",header=None, on_bad_lines='warn')
    except FileNotFoundError:
        print(f"No .dlm file found in the directory entered")
    else:
        print(f"File {i+1}: {filename[-19:]}\n", end=' ')
        
    raw.columns = col_names
        
    # Clear original time data stored in the first row
    raw.loc[0,'time'] = 0
    # data error results in NA values in epochNum, exclude rows with NA
    raw.dropna(inplace=True)
    # rows with epochNum == NA may have non-numeric data recorded. In this case, change column types to float for calculation. not necessary for most .dlm.
    raw[['time','col2','col3','col4','col5','col6','col7']] = raw[['time','col2','col3','col4','col5','col6','col7']].astype('float64',copy=False)

    return raw


#%%

root = '/Volumes/ZhuBackup/biolum/250514/250514.0001'
# find dlm files in the current directory 
# assign to filenames
filenames = glob.glob(os.path.join(root,"*.dlm"))
for i, file in enumerate(filenames):
    raw = read_dlm(i, file)
# %%
