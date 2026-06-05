import numpy as np
import pandas as pd
from sklearn import metrics
import math


def cohensd_alt(d1, d2):
    # https://aakinshin.net/posts/nonparametric-effect-size2/
    # https://aakinshin.net/posts/nonparametric-effect-size/
    # d1 and d2 shoule be RAW data
    if type(d1) == pd.core.series.Series:
        d1 = d1.values
        d2 = d2.values
    CONSISTENCY_CONSTANT = 1.4826
    d1_median = np.median(d1)
    d2_median = np.median(d2)
    d1_adj = np.abs([val-d1_median for val in d1])
    d2_adj = np.abs([val-d2_median for val in d2])
    MAD1 = CONSISTENCY_CONSTANT * np.median(d1_adj) 
    MAD2 = CONSISTENCY_CONSTANT * np.median(d2_adj) 
    n1, n2 = len(d1), len(d2)
    pmad = math.sqrt(((n1 - 1) * MAD1**2 + (n2 - 1) * MAD2**2) / (n1 + n2 - 2))
    return (d1_median - d2_median) / pmad


def cohen_d_original(d1, d2):
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    v1, v2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

def print_values(df, if_normal=False, features=None, feature_name=None, nd=2):
    if features:
        if type(features) == str:
            features = [features]
        for col in features:
            if not feature_name:
                feature_name = features
            print(f"*** {col} ***")
            values = df[col].values
            if if_normal:
                mu = np.nanmean(values)
                sd = np.nanstd(values)
                print(f"\def\{feature_name}MEAN{{{mu:.{nd}f}}} \n\def\{feature_name}SD{{{sd:.{nd}f}}}")
            else:
                mu = np.nanmedian(values)
                iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
                print(f"\def\{feature_name}Median{{{mu:.{nd}f}}} \n\def\{feature_name}IQR{{{iqr:.{nd}f}}}")
    else:
        if not feature_name:
            feature_name = features
        values = df
        if if_normal:
            mu = np.nanmean(values)
            sd = np.nanstd(values)
            print(f"\def\{feature_name}MEAN{{{mu:.{nd}f}}} \n\def\{feature_name}SD{{{sd:.{nd}f}}}")
        else:
            mu = np.nanmedian(values)
            iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
            print(f"\def\{feature_name}MEDIAN{{{mu:.{nd}f}}} \n\def\{feature_name}IQR{{{iqr:.{nd}f}}}")

def calc_ROC(data,feature,ctrl_name, condition_col, chg_dir):
    '''
    data: long format dataframe containing feature to calculate and both cond and ctrl data
    feature: col name of the col in data to plot
    ctrl_name: name of the control in the condition column
    '''
    if chg_dir == 'increase':
        pos_label = 0
    else:
        pos_label = 1
        
    all_cond = list(set(data[condition_col])).sort()
    y_true = data[condition_col].map({all_cond[0]:1,all_cond[1]:0})
    y_test = data[feature]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_test, pos_label = pos_label)
    
    # jackknife to calculate auc variance
    ctrl_all = data.loc[data[condition_col]==ctrl_name,feature].values
    cond_all = data.loc[data[condition_col]!=ctrl_name,feature].values
    auc = []
    for jack in range(len(ctrl_all)-1):
        ctrl = np.delete(ctrl_all,jack)
        cond = np.delete(cond_all,jack)
        y_test = np.concatenate((ctrl,cond))
        y_true = np.concatenate([np.repeat(1,len(ctrl)),np.repeat(0,len(cond))])
        fpr_jack, tpr_jack, _ = metrics.roc_curve(y_true, y_test, pos_label = pos_label)
        auc.append(metrics.auc(fpr_jack, tpr_jack))                       
    
    return fpr, tpr, auc


# %% bootstrap confidence interval for any statistical parameter
# https://aegis4048.github.io/non-parametric-confidence-interval-with-bootstrap

def bootstrap_simulation(sample_data, num_realizations):
    n = sample_data.shape[0]
    boot = []
    for i in range(num_realizations):
        real = np.random.default_rng().choice(sample_data.values.flatten(), n, replace=True) 
        boot.append(real)
    columns = ['Real ' + str(i + 1) for i in range(num_realizations)]
    return pd.DataFrame(boot, index=columns).T

def calc_sum_stats(boot_df):
    sum_stats = boot_df.describe().T[['mean', 'std', 'min', 'max']]
    sum_stats['median'] = boot_df.median()
    sum_stats['skew'] = boot_df.skew()
    # sum_stats['kurtosis'] = boot_df.kurtosis()
    sum_stats['IQR'] = boot_df.quantile(0.75) - boot_df.quantile(0.25)
    return sum_stats.T
 
def calc_bounds(conf_level):
    assert (conf_level < 1), "Confidence level must be smaller than 1"
    margin = (1 - conf_level) / 2
    upper = conf_level + margin
    lower = margin
    return margin, upper, lower

def calc_confidence_interval(df_sum_stats, conf_level): 
    margin, upper, lower = calc_bounds(conf_level)
    conf_int_df = df_sum_stats.T.describe(percentiles=[round(lower,4), 0.5, round(upper,4)]).iloc[4:7, :].T
    conf_int_df.columns = ['P' + str(round(lower * 100, 1)), 'P50', 'P' + str(round(upper * 100, 1))]
    return conf_int_df 

def print_confidence_interval(conf_df, conf_level):
    print('By {}% chance, the following statistics will fall within the range of:\n'.format(round(conf_level * 100, 1)))
    margin, upper, lower = calc_bounds(conf_level)
    upper_str = 'P' + str(round(upper * 100, 1))
    lower_str = 'P' + str(round(lower * 100, 1))
    for stat in conf_df.T.columns:
        lower_bound = round(conf_df[lower_str].T[stat], 3)
        upper_bound = round(conf_df[upper_str].T[stat], 3)
        mean = round(conf_df['P50'].T[stat], 3)
        print("{0:<10}: {1:>10}  ~ {2:>10} , AVG = {3:>5}".format(stat, lower_bound, upper_bound, mean))

def boot_ci(data, stat, boot_rep=500, confidence_level=0.9, if_print=False):
    '''
    Calculate lower and upper CI boundaries at given confidence level
    to use, define a new function defining your parameter of interest and desired confidence level:
        plt_median_ci = partial(boot_ci, stat='median', boot_rep=100, confidence_level=0.99)
    '''
    boot_data = bootstrap_simulation(data, boot_rep)
    boot_sum_stats = calc_sum_stats(boot_data)
    conf_int = calc_confidence_interval(boot_sum_stats, confidence_level)
    if if_print:
        print_confidence_interval(conf_int, confidence_level)
    lower = conf_int.iloc[:,0][stat]
    upper = conf_int.iloc[:,2][stat]
    return [lower, upper]
