
#%%
import os
from pathlib import Path
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import (set_font_type)
from tqdm import tqdm
from datetime import datetime
import math

# %%
def grp_by_epoch_copy(df):
    # Group df by 'epochNum'. If cols is empty take all columns
    return df.groupby('epochNum', sort=False)

def smooth_series_ML_copy(a,WSZ):
    '''
    Modified from Divakar's answer https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    a: NumPy 1-D array containing the data to be smoothed
    WSZ: smoothing window size needs, which must be odd number,
    as in the original MATLAB implementation
    '''
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    res = np.concatenate((  start , out0, stop  ))
    return pd.Series(data=res, index=a.index)

def read_dlm_copy(i,filename):
    # read_dlm takes file index: i, and the file name end with .dlm
    col_names = ['time','fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']
    try:
        # raw = pd.read_csv(filename, sep="\t",names = col_names) # load .dlm
        raw = pd.read_csv(filename, sep="\t",header=None)
    except FileNotFoundError:
        print(f"No .dlm file found in the directory entered")
    else:
        print(f"File {i+1}: {filename[-19:]}", end=' ')
    
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
    raw[['fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']] = raw[['fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']].astype('float64',copy=False)
    return raw

# %%
'''
Plots single epoch that contains one or more bouts
Input directory needs to be a folder containing analyzed dlm data.
'''
print('\n- Plotting time series (raw)')

root = "/Volumes/LabData/SAMPL_data_in_use/zhu_2022_method/data_for_movie/selected O3/220620 9dpf video"
FRAME_RATE = 166
SCALE = 60
# %% features for plotting
# below are all the properties can be plotted. 
all_features = {
    'ang':'pitch (deg)',
    # 'absy':'y position (mm)'
    # 'deltaT',
    # 'x', 'y',
    'headx':'head x (mm)',
    'heady':'head y (mm)',
    'centeredAng':'centered angle (deg)',
    # 'xvel', 
    # 'yvel', 
    'dist':'distance (mm)',
    'displ':'displacement (mm)',
    'angVel':'ang vel (deg*s-1)',
    # 'angVelSmoothed', 
    'angAccel':'ang accel (deg*s-2)',
    'swimSpeed':'speed (mm*s-1)',
    'velocity':'velocity (mm*s-1)'
}
# %%
# generate figure folder
folder_name = 'E_timeseries_raw'
fig_dir = "/Users/yunluzhu/Documents/Lab2/Manuscripts/2022-07_Behavior_apparatus/movie_figures"

try:
    os.makedirs(fig_dir)
except:
    print('Figure folder already exist! Old figures will be replaced.')

# %%
# get all dlm
all_dlm = [file for file in os.listdir(root) if file.endswith(".dlm") ]
if len(all_dlm) > 1:
    raise Exception('only one dlm file allowed')

# %%
file = os.path.join(root,all_dlm[0])
raw = read_dlm_copy(0,file)
# %%
XY_SM_WSZ = 9  # smooth window size for x and y coordinates

raw['absx'] = smooth_series_ML_copy(raw.loc[:,'absx'],XY_SM_WSZ)
raw['absy'] = smooth_series_ML_copy(raw.loc[:,'absy'],XY_SM_WSZ)

ana = raw.assign(
    deltaT = grp_by_epoch_copy(raw).time.diff()
)
# Get the start time from file name
datetime_frmt = '%y%m%d %H.%M.%S'
time_stamp = file[-19:-4]
start_time = datetime.strptime(time_stamp, datetime_frmt)
# Calculate absolute datetime for each timepoint. DataFrame calculation is faster than using .apply()
ana.insert(1,
    'absTime', start_time + pd.to_timedelta(raw['time'].values, unit=('s'))
)

# Calculate coordinates
#   dataframe subduction is faster than cumulative difference. 
#   Use .values to convert into arrays to further speed up
centered_coordinates = pd.DataFrame((
    raw[['absx','absy','absHeadx','absHeady','ang']].values
    - grp_by_epoch_copy(raw)[['absx','absy','absHeadx','absHeady','ang']].transform('first').values
), columns = ['x','y','headx','heady','centeredAng'])

ana = ana.join(centered_coordinates)

ana_g = grp_by_epoch_copy(ana)

ana = ana.assign(
    # x and y velocity. using np.divide() has shorter runtime than df.div()
    xvel = np.divide(ana_g['x'].diff().values, ana['deltaT'].values),
    yvel = np.divide(ana_g['y'].diff().values, ana['deltaT'].values),
    # use numpy function np.linalg.norm() for displacement and distance
    dist = np.linalg.norm(ana_g[['x','y']].diff(), axis=1),
    # since beginning coordinates for each epoch has been set to 0, just use (x, y) values for displ
    displ = ana.groupby('epochNum', as_index=False, sort=False).apply(
        lambda g: pd.Series((np.linalg.norm(g[['x','y']], axis=1)),index = g.index).diff()
    ).reset_index(level=0, drop=True),
    # array calculation is more time effieient
    angVel = np.divide(ana_g['ang'].diff().values, ana['deltaT'].values)
)
# now let's get smoothed angular vel and angular acceleration
ana = ana.assign(  
    # loop through each epoch, get second to last angVel values (exclude the first one which is NA)
    # calculate smooth, keep the index, assign to the new column
    # angVelSmoothed = pd.concat(
    #     smooth_series_ML_copy(g.tail(len(g)-1)['angVel'],3) for i, g in grp_by_epoch_copy(ana)
    # ),
    angAccel = np.divide(grp_by_epoch_copy(ana)['angVel'].diff().values, ana['deltaT'].values),
)
# %%
res = ana
res.loc[:,['y','heady','yvel']] = res[['y','heady','yvel']] * -1 / SCALE
res.loc[:,['x','headx','xvel','displ','dist','fishLen']] = res[['x','headx','xvel','displ','dist','fishLen']] / SCALE
# calculate swim speed
res.loc[:,'swimSpeed'] = np.divide(res['dist'].values, res['deltaT'].values)
# calculate swim velocity (displacement/)
res.loc[:,'velocity'] = np.divide(res['displ'].values, res['deltaT'].values)
# define fish length as 70th percentile of lengths captured.
fish_length = grp_by_epoch_copy(res)['fishLen'].agg(
    fishLenEst = lambda l: l.quantile(0.7)
).reset_index()

# %%
all_epoch_lengths = res.groupby('epochNum').size().reset_index()
all_epoch_lengths.columns = ["epochNum", 'frames']
all_epoch_lengths = all_epoch_lengths.assign(
    duration = all_epoch_lengths['frames']/166
)
all_epoch_lengths.sort_values(by='duration',ascending=False)



# %%
res = res.assign(
    # time_point = [str(tp) for tp in res['absTime']]
    day = [tp.day for tp in res['absTime']],
    hour = [tp.hour for tp in res['absTime']],
    min = [tp.minute for tp in res['absTime']],
    sec = [tp.second for tp in res['absTime']],
    
)
# video_file_name = input('video file name: ')
video_file_name = '220620 14.12.36-163492'
day_match = round_half_up(video_file_name[4:6])
hour_match = round_half_up(video_file_name[7:9])
min_match = round_half_up(video_file_name[10:12])
sec_match = round_half_up(video_file_name[13:15])

matched_epoch = res.loc[(res['hour']==hour_match) & 
                     (res['min']==min_match) & 
                    #  (res['sec']==sec_match) & 
                     (res['day']==day_match)]

# matched_epoch_g = matched_epoch.groupby('epochNum').size()
# print(f"{len(matched_epoch_g)} epoch found. Duration = {matched_epoch_g.values/166}")

matched_epoch_previousMIN = res.loc[(res['hour']==hour_match) & 
                     (res['min']==min_match-1) & 
                    #  (res['sec']==sec_match) & 
                     (res['day']==day_match)]

# %%
all_timepoints = pd.concat([matched_epoch_previousMIN,matched_epoch])

all_epochs = pd.Series(list(set(all_timepoints['epochNum'])))
epochs_cut = np.arange(all_epochs.min(),all_epochs.max(),10)

all_timepoints = all_timepoints.assign(
    epoch_group = pd.cut(all_timepoints['epochNum'],np.append(epochs_cut,np.inf),labels = np.arange(len(epochs_cut)))
)
all_timepoints_f = all_timepoints.loc[all_timepoints['epoch_group']==4]
# %%
all_timepoints_ff = all_timepoints_f = all_timepoints.loc[all_timepoints['epochNum']==179]

sns.relplot(data = all_timepoints_ff,
             x = 'absHeadx',
             y = 'absHeady',
             alpha = 1,
             hue = 'epochNum',
             kind='scatter'
             )
plt.xlim(0, 1200)
plt.ylim(0, 1200)

# %%
# sns.scatterplot(data = all_timepoints_ff,
#              x = 'time',
#              y = 'ang',
#                )
data_toplt = all_timepoints_ff.loc[:,all_features.keys()]
data_toplt = data_toplt.rename(columns=all_features)
        
data_toplt = data_toplt.assign(
    epochNum = all_timepoints_ff['epochNum'].values,
    deltaT = all_timepoints_ff['deltaT'].values
)
data_toplt = data_toplt.reset_index(drop=True)

EPOCH_BUF = math.ceil(FRAME_RATE*0.2)      
data_toplt = data_toplt.loc[(data_toplt.index >= EPOCH_BUF) &
                            (data_toplt.index <= max(data_toplt.index)-EPOCH_BUF)]

data_toplt = data_toplt.assign(
    time_ms = np.cumsum(data_toplt['deltaT'])*1000
)


# %%


set_font_type()

for feature_toplt in tqdm(list(all_features.values())):
    p = sns.relplot(
        data = data_toplt, x = 'time_ms', y = feature_toplt,
        kind = 'line',aspect=3, height=2
        )
    plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_raw.pdf"),format='PDF')





