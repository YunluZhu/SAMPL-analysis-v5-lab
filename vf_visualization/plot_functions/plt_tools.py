import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

def set_font_type():
    mpl.rcParams['pdf.fonttype'] = 42
    
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

def day_night_split(df,time_col_name,**kwargs):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    day_index = hour[(hour>=9) & (hour<23)].index
    night_index = hour[(hour<9) | (hour>=23)].index

    df = df.assign(ztime = 'day')
    which_ztime = 'day'
    
    for key, value in kwargs.items():
        if key == 'ztime':
            which_ztime = value
    
    if which_ztime == 'day':
        df_out = df.loc[day_index, :]
    elif which_ztime == 'night':
        df_out = df.loc[night_index, :]
        df_out.loc[night_index,'ztime'] = 'night'
    elif which_ztime == 'all':
        df_out = df
        df_out.loc[night_index,'ztime'] = 'night'
    return df_out, day_index, night_index

def distribution_binned_average(df, by_col, bin_col, bin):
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins)
    df_out = grp[[by_col,bin_col]].mean()
    return df_out