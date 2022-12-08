'''
Read one .dlm file and return a dataframe
Modified from:
analyzeFreeVerticalGrouped2.m by DEE 1.30.2015
    "the LabView code returns a value to mark an "epoch," which is a continuous series of frames that had at least one identified particle 
+ lines to output head location in addition to body, for detection direction of movement." 
'''
# %%
import pandas as pd
import numpy as np
import seaborn as sns
# %%
filename = '/Users/yunluzhu/Documents/Lab2/Data/SAMPL_tmp/220402 Ximea reduced vid size/220402 12.37.15.dlm'

# %%
# read_dlm takes file index: i, and the file name end with .dlm
# file_path = os.getcwd() # need to be defined/entered when calling as a function or use command line input
# filenames = os.listdir
# filenames = glob.glob(f"{file_path}/*.dlm") # subject to change 
col_names = ['time','fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']

# raw = pd.read_csv(filename, sep="\t",names = col_names) # load .dlm
raw = pd.read_csv(filename, sep="\t",header=None)


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

# %%
tmp = raw.groupby('epochNum').apply(
    lambda x: np.diff(x['time'])
)
all_time_itv = np.concatenate(tmp)

# %%
to_plt = np.random.choice(all_time_itv,5000,replace=False)
sns.set(rc={'figure.figsize':(8,4)})
sns.histplot(to_plt,bins=50)
# %%
to_plt = np.random.choice(all_time_itv,5000,replace=False)
sns.kdeplot(to_plt)

# %%
