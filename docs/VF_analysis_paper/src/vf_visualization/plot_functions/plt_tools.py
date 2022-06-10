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

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    return df_day