import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import scipy.stats as st



def extract_consecutive_bout_features(connected_bout_df:pd.DataFrame, list_of_features:list, max_lag:int):
    """get consecutive bout features by cond1 and cond2, return in long format - YZ 230502
    NOTE cross-condition function

    Args:
        connected_bout_df (pd.DataFrame): output from get_connected_bouts()
        list_of_features (list): a list containing features/columns to keep/extract 
        max_lag (int): maximun number of "lags", unit = bout. e.g. if 2, then extract all serieses of 3 bouts

    Returns:
        pd.DataFrame: Long format of consecutive bouts numbered by "lag" and id'd by "id"
        pd.DataFrame: a copy of the input bout features dataframe, with 2 new columns: epoch_conduid and exp_conduid

    """

    connected_bout_df = connected_bout_df.assign(
        epoch_conduid = connected_bout_df['cond0'] + connected_bout_df['cond1'] + connected_bout_df['expNum'].astype(str) + connected_bout_df['epoch_uid'],
        exp_conduid = connected_bout_df['cond0'] + connected_bout_df['cond1'] + connected_bout_df['expNum'].astype(str),
    )
    connected_bout_df = connected_bout_df.groupby(['epoch_conduid']).filter(lambda g: len(g)>1)

    consecutive_bout_features = pd.DataFrame()

    for feature_toplt in list_of_features:
        shifted_df = pd.DataFrame()
        autoCorr_res = pd.DataFrame()
        df_tocorr = pd.DataFrame()

        grouped = connected_bout_df.groupby(['cond1','cond0','expNum'])
        col_selected = feature_toplt
        for (cond1, cond0, expNum), group in grouped:
            shift_df = pd.concat([group[col_selected].shift(-i).rename(f'{col_selected}_{i}') for i in range(max_lag+1)], axis=1)
            this_df_tocorr = shift_df.groupby(group['epoch_conduid']).apply(
                lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
            )
            this_df_tocorr = this_df_tocorr.assign(
                cond1=cond1, 
                cond0=cond0,
                expNum=expNum,
                exp_conduid=group['exp_conduid']
                )
            df_tocorr = pd.concat([df_tocorr, this_df_tocorr], ignore_index=True)

        sel_bouts = df_tocorr
        sel_bouts = sel_bouts.loc[sel_bouts[f'{feature_toplt}_{max_lag}'].notna()]
        sel_bouts = sel_bouts.assign(
            first_bout = sel_bouts[f'{feature_toplt}_0'],
        )
        sel_bouts['id'] = sel_bouts.index
        long_df = pd.wide_to_long(sel_bouts, stubnames=feature_toplt, sep='_', j='lag', i='id').reset_index()
        long_df = long_df.loc[long_df['lag']<=max_lag]
        
        long_df = long_df.rename(columns={'first_bout': f'{feature_toplt}_first'})
        
        if consecutive_bout_features.empty:
            consecutive_bout_features = long_df
        else:
            consecutive_bout_features = consecutive_bout_features.merge(long_df, on=['id', 'lag','cond1','cond0','expNum','exp_conduid'], how='left')
    return consecutive_bout_features, connected_bout_df

def cal_autocorrelation_feature(this_cond_df:pd.DataFrame, col_selected:str, col_groupby:str, max_lag:int):
    """calculate autocorrelation and slope of auto-linearRegression for consecutive bouts with different lags/intervals. -YZ 230502
    NOTE single-condition function

    Args:
        this_cond_df (pd.DataFrame): output from get_connected_bouts(). NOTE separate df by conditions before running this func if intended 
        col_selected (str): feature/column to be calculated
        col_groupby (str): feature/column to be grouped by
        max_lag (int): maximun number of "lags" for autocorrelation, unit = bout. e.g. if 2, then extract all serieses of 3 bouts

    Returns:
        pd.DataFrame: autocorrelation/regression results
        pd.DataFrame: Long format of shifted consecutive bout feature dataframe
        pd.DataFrame: Wide format of shifted consecutive bout feature dataframe for correlation calculation
    """
    
    if ("epoch_conduid" not in this_cond_df.columns) | ("exp_conduid" not in this_cond_df.columns):
        this_cond_df = this_cond_df.assign(
            epoch_conduid = this_cond_df['cond0'] + this_cond_df['cond1'] + this_cond_df['expNum'].astype(str) + this_cond_df['epoch_conduid'],
            exp_conduid = this_cond_df['cond0'] + this_cond_df['cond1'] + this_cond_df['expNum'].astype(str),
        )
    
    slope = []
    slope_err=[]
    corrres = []
    pearsonRci = []
    lag = []
    n = []
    intercept = []

    long_form_shifted = pd.DataFrame()
    shift_df = pd.concat([this_cond_df[col_selected].shift(-i).rename(f'{col_selected}_{i}') for i in range(max_lag+1)], axis=1)
    df_to_corr = shift_df.groupby(this_cond_df[col_groupby]).apply(
        lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
    )
    for j in np.arange(1,max_lag+1):
        this_df = df_to_corr.iloc[:,[0,j]].dropna(axis='rows')
        if len(this_df[this_df[f'{col_selected}_{j}'].notna()]) >= 10:
            x = this_df[f'{col_selected}_0']
            y = this_df[f'{col_selected}_{j}']
            this_corr = st.pearsonr(x, y)
            # regression_model = LinearRegression()
            # regression_model.fit(x.values.reshape(-1,1), y)
            this_slope, this_intercept, this_r, this_p, this_se = st.linregress(x, y)
            intercept.append(this_intercept)
            slope.append(this_slope)
            slope_err.append(this_se)
            corrres.append(this_corr[0])
            pearsonRci.append(this_corr.confidence_interval())
            lag.append(j)
            n.append(len(this_df[this_df[f'{col_selected}_{j}'].notna()]))
            
            this_df.columns = ['ori', 'shifted']
            long_form_shifted = pd.concat([long_form_shifted, this_df.assign(lag=j)], ignore_index=True)
    output = pd.DataFrame(data={
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        f'autocorr_{col_selected}': corrres,
        'lag': lag,
        'ci': [[np.abs(ci[0] - corrres[i]), ci[1] - corrres[i]] for i, ci in enumerate(pearsonRci)],
        'n': n,
    })
    return output, long_form_shifted, df_to_corr

