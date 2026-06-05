import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from decimal import Decimal
import decimal


def MAD(values):
    if len(values.shape) == 1:
        return np.median(np.absolute(values - np.nanmedian(values)))
    else:
        tmp = np.absolute(values - values.groupby(np.ones(len(values))).transform(np.median))
        return tmp.median()

def jackknife_avg2(df, cat_col, jackknife_col, data_col, sampling=True, method='mean'):

    if type(data_col) == str:
        data_col = [data_col]
    if type(cat_col) == str:
        cat_col = [cat_col]
    if sampling:
        median_num_resample = df.groupby(cat_col).size().median()
        df = df.groupby(cat_col).sample(int(median_num_resample), replace=True)
    new_jackknife_idx = np.array([])
    res = pd.DataFrame()
    for cat, group in df.groupby(cat_col):
        for i in group[jackknife_col].unique():
            if method == 'mean':
                this_res = group.loc[group[jackknife_col] != i, data_col].mean().to_frame().T.assign()
            elif method == 'median':
                this_res = group.loc[group[jackknife_col] != i, data_col].median().to_frame().T.assign()
            for j, category in enumerate(cat_col):
                this_res = this_res.assign(new_cat = cat[j])
                this_res = this_res.rename(columns={'new_cat':category})
            res = pd.concat([res, this_res], ignore_index=True)
        new_jackknife_idx = np.append(new_jackknife_idx,group[jackknife_col].unique())
                
    res = res.assign(
        jackknife_col = new_jackknife_idx
    )
    res = res.rename(columns={'jackknife_col':jackknife_col})
    return res


def get_2sd(var:list):
    n=2
    sdval = np.nanstd(var) 
    meanval = np.nanmean(var)
    return(meanval-sdval*n, meanval+sdval*n)

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

def defaultPlotting(**kwargs):
    font_size = 12
    for key, value in kwargs.items():
        if key == 'size':
            font_size = round_half_up(value)
    sns.set(rc={"xtick.labelsize":font_size,"ytick.labelsize":font_size, "axes.labelsize":font_size},style="ticks")

def jackknife_list(ori_list):
    matrix = np.tile(ori_list,(len(ori_list),1))
    output = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return output

# def jackknife_mean(df):
#     output = pd.DataFrame()
#     for i in list(df.index):
#         output = pd.concat([output, df.loc[df.index != i,:].\
#             mean().to_frame().T])
#     return output

def jackknife_mean_by_col(df,col,method='mean'):
    output = pd.DataFrame()
    all_repeats = list(set(df[col]))
    for i in all_repeats:
        if method == 'mean':
            this_mean = df.loc[df[col] != i,:].mean(numeric_only=True).to_frame().T
        if method == 'median':
            this_mean = df.loc[df[col] != i,:].median(numeric_only=True).to_frame().T
        output = pd.concat([output, this_mean.assign(jackknife_idx=i)])
    return output


def day_night_split(df,time_col_name, narrow_bin = False, **kwargs):
    which_ztime = 'day'
    for key, value in kwargs.items():
        if key == 'ztime':
            which_ztime = value
            
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    
    if narrow_bin:
        day_night_idx = pd.cut(hour,
                        bins=[-1, 7, 9, 21, 24],  # Define edges explicitly
                        labels=['night', 'transition1', 'day', 'transition2'],  
                        ordered=True
                        )
    else:
        day_night_idx = pd.cut(hour,[-1,8,22,24],labels=['night','day','night2']) # pd.cut() doesn't support duplicated labels
        
    day_night_idx.loc[day_night_idx=='night2'] = 'night'
    df = df.assign(ztime = list(day_night_idx))
    
    if which_ztime == 'all':
        df_out = df
    else:
        df_out = df.loc[df['ztime']==which_ztime, :]
    return df_out#, day_index, night_index


def distribution_binned_average(df, by_col, bin_col, bin, method='mean'):
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins, observed=True)
    if method == 'mean':
        df_out = grp[[by_col,bin_col]].mean()
    elif method == 'median':
        df_out = grp[[by_col,bin_col]].median()
    elif method == 'std':
        df_out = grp[[by_col,bin_col]].std()
    elif method == 'mad':
        df_out = grp[[by_col,bin_col]].apply(lambda x: MAD(x))
    return df_out


def distribution_binned_average_opt(df, by_col, bin_col, bin, method='mean'):
    bins = pd.cut(df[by_col], bins=bin)
    agg_func = 'mean' if method == 'mean' else 'median'
    df_out = df.groupby(bins, observed=False)[[by_col, bin_col]].agg(agg_func)
    return df_out


def distribution_binned_sum(df, by_col, bin_col, bin):
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins)
    df_out = grp[[bin_col]].sum()
    df_out[by_col] = grp[[by_col]].mean()
    return df_out

def linReg_sampleSatter_plot(data,xcol,ycol,xmin,xmax,color):
    xdata = data[xcol] 
    ydata = data[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    x = np.linspace(xmin,xmax,100)
    y = slope*x+intercept
    plt.figure(figsize=(4,4))
    g = sns.scatterplot(x=xcol, 
                        y=ycol, 
                        data=data.sample(2000), 
                        # marker='+',
                        alpha = 0.1,
                        color='grey',
                        edgecolor="none",
                        )
    plt.plot(x, y, color=color)
    return g, slope, intercept, r_value, p_value, std_err


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