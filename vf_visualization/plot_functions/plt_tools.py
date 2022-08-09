import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

def set_font_type():
    mpl.rcParams['pdf.fonttype'] = 42
    
def defaultPlotting(**kwargs):
    font_size = 12
    for key, value in kwargs.items():
        if key == 'size':
            font_size = int(value)
    sns.set(rc={"xtick.labelsize":font_size,"ytick.labelsize":font_size, "axes.labelsize":font_size},style="ticks")

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

def jackknife_mean_by_col(df,col):
    output = pd.DataFrame()
    all_repeats = list(set(df[col]))
    for i in all_repeats:
        this_mean = df.loc[df[col] != i,:].mean(numeric_only=True).to_frame().T
        output = pd.concat([output, this_mean.assign(jackknife_idx=i)])
    return output

def day_night_split(df,time_col_name,**kwargs):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    # day_index = hour[(hour>=9) & (hour<23)].index
    # night_index = hour[(hour<9) | (hour>=23)].index
    day_night_idx = pd.cut(hour,[-1,8,22,24],labels=['night','day','night2']) # pd.cut() doesn't support duplicated labels
    day_night_idx.loc[day_night_idx=='night2'] = 'night'
    df = df.assign(ztime = list(day_night_idx))
    which_ztime = 'day'
    
    for key, value in kwargs.items():
        if key == 'ztime':
            which_ztime = value
    
    if which_ztime == 'all':
        df_out = df
    else:
        df_out = df.loc[df['ztime']==which_ztime, :]
    return df_out#, day_index, night_index

def distribution_binned_average(df, by_col, bin_col, bin):
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins)
    df_out = grp[[by_col,bin_col]].mean()
    return df_out