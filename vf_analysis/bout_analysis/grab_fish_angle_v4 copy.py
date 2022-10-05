'''
Modified from DEE's GrabFishAngelAll.m
Functions:
    1. Select qualified epochs
    2. Select qualified bouts
    3. Extract bout properties, align by peak speed
    4. Calculate other properties, such as speed, displacement, acceleration...

Output:
    1. 3 hdf5 files, including all dataframes.
    2. 1 catalog csv file.

To use, run the grab_fish_angle.run() , which calls grab_fish_angle.grab_fish_angle()

NOTE
220505: shortened bout align window
220718: IEI detection changed to 100ms after and 100ms before speed passes speed threshold. Speed threshold set to 3.5mm/s
220826: max angular acceleration filter now applys to moving average (window 3) of angaccel. filtering out about 10% epochs per dlm.
220826: added logging function
'''
# %%
# Import Modules and functions
import sys
import os
import glob
import configparser
import pandas as pd # pandas library
import numpy as np # numpy
from collections import defaultdict
import time
from datetime import datetime
from datetime import timedelta
import math
from preprocessing.read_dlm import read_dlm
from preprocessing.analyze_dlm_v4 import analyze_dlm_resliced
from bout_analysis.logger import log_vf_ana

global program_version
program_version = 'V4.3.220826'


# %%
# Define functions

def read_parameters(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    box_number = config.getint('User-defined parameters','Box number')
    genotype = config.get('User-defined parameters','Genotype').replace('"','')
    age = config.getint('User-defined parameters','Age')
    notes = config.get('User-defined parameters','Notes').replace('"','')
    initials = config.get('User-defined parameters','Inititals').replace('"','')
    light_cycle = config.getint('User-defined parameters','Light cycle')
    dir = config.get('User-defined parameters','Save data to?').replace('"','')
    line_1 = config.getint('User-defined parameters','Mom line number')
    line_2 = config.getint('User-defined parameters','Dad line number')
    cross_id = config.get('User-defined parameters','cross ID').replace('"','')
    num_fish = config.getint('User-defined parameters','Num fish')
    filename = config.get('User-defined parameters','Filename').replace('"','')
    parameters = pd.DataFrame({
        'box_number':box_number,
        'genotype':genotype,
        'age':age,
        'notes':notes,
        'initials':initials,
        'light_cycle':light_cycle,
        'dir':dir,
        'line_1':line_1,
        'line_2':line_2,
        'cross_id':cross_id,
        'num_fish':num_fish,
        'filename':filename,
    }, index=[0])
    return parameters


def grp_by_epoch(df):
    '''Group df by 'epochNum'''
    return df.groupby('epochNum', sort=False)

def grp_by_swim(df,loco_index):
    '''Get bouts, then group by 'col'''
    return df.loc[df[loco_index] % 2 == 1].groupby(loco_index, as_index=False, sort=False)

def smooth_series_ML(a,WSZ):
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

def smooth_ML(a,WSZ):
    '''
    Modified from Divakar's answer https://stackoverflow.com/questions/404430
    /matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    a: NumPy 1-D array containing the data to be smoothed
    WSZ: smoothing window size needs, which must be odd number,
    as in the original MATLAB implementation
    '''
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    res = np.concatenate((  start , out0, stop  ))
    return res

# %%
# # Read analyzed epoch files

# # subjected to change
# file_path = os.getcwd() # need to be defined/entered when calling as a function or use command line input
# filenames = glob.glob(f"{output_dir}/*_analyzed_epochs.pkl") # subject to change
# file_i = 0 # get 1 file for testing

# analyzed = pd.read_pickle(filenames[file_i])
# fish_length = pd.read_pickle(f"./data/{file_i+1}_fish_length.pkl")

def grab_fish_angle(analyzed, fish_length,sample_rate):
    '''
    Function to analyze epochs, find bouts, and calculate things we care
    Output: one dictionary with multiple dataframes, each column in each dataframe is equivalent to one output file in the original MATLAB code
    '''
    # %%
    # Constants
    PROPULSION_THRESHOLD = 4  # mm/s, speed threshold above which samples are considered propulsion
    BASELINE_THRESHOLD = 2  # mm/s, speed threshold below which samples are considered at baseline (not propelling)
    SAMPLE_RATE = sample_rate  # Hz
    MIN_SWIM_INTERVAL = 0.1  # s, minimum swim interval duration
    POST_BOUT_BUF = math.ceil(0.1 * SAMPLE_RATE)  # 4 frames for 40hz
    PRE_BOUT_BUF = math.ceil(0.1 * SAMPLE_RATE)  # 4 frames for 40hz
    BOUT_WINDOW_HALF = math.ceil(0.3 * SAMPLE_RATE)  # 12 frames for 40hz, changed in v4
    # SM_WINDOW = math.ceil((3/40) * SAMPLE_RATE)  # 3 for 40hz
    SM_WINDOW = 3
    SPD_SM_WINDOW = math.ceil((1/40) * SAMPLE_RATE) # smooth speed for bout detection

    # correct smooth windows, must be odd
    SM_WINDOW = SM_WINDOW-1+SM_WINDOW%2
    SPD_SM_WINDOW = SPD_SM_WINDOW-1+SPD_SM_WINDOW%2

    EDGE_CHOP = math.ceil((2.5/40) * SAMPLE_RATE)   # number of samples to remove from the beginning and end of each vector to account for edge effects (improper detection of fish body and movement)
    BOUT_LONG_TAIL = SAMPLE_RATE  # align prop bouts with longer duration (20 extra frames for 40hz)

    PRE_PEAK_FRAMES = math.ceil(SAMPLE_RATE * 0.5)  # s, Only align bouts with extra frames before peak speed
    POST_PEAK_FRAMES = math.ceil(SAMPLE_RATE * 0.3)  # s, Only align bouts with extra frames after peak speed
    All_Aligned_FRAMES = PRE_PEAK_FRAMES+POST_PEAK_FRAMES+1
    IEI_tail = math.ceil(SAMPLE_RATE * 0.5)
    IEI_2_swim_buf = math.ceil(0.1 * SAMPLE_RATE)



    # bout index for aligned bouts
    bout_idx_peak = PRE_PEAK_FRAMES # idx of peak after alignment
    frame_number25 = int(0.025*SAMPLE_RATE)
    frame_number100 = int(0.1*SAMPLE_RATE) # 100ms, 4 at 40Hz
    frame_number250 = int(0.25*SAMPLE_RATE) # 250ms, 10 at 40Hz
    frame_number125 = int(0.125*SAMPLE_RATE) # 125ms, 5 at 40Hz

    # find epochs with max speed above threshold
    df = grp_by_epoch(analyzed).filter(
        lambda g: np.nanmax(g['swimSpeed'].values) >= PROPULSION_THRESHOLD
    ).reset_index(drop=True)
    
    if len(df) < 3:    # if no epoch found
        return "> not enough epoches > dlm file skipped"
    
    
    grouped_df = grp_by_epoch(df)

    # calculate smoothed x and y velocity for heading calculation later. Save into a new dataaframe, which will be chopped for heading matched calculations later
    df_chopped = df.assign(
        # because SM_WINDOW = 3 and the first vel of each epoch is NA
        # and NA values caused by smoothing will be dropped later, smooth through all the rows won't cause epoch-crossing troubles
        xvel_sm = smooth_series_ML(df['xvel'],SM_WINDOW),
        yvel_sm = smooth_series_ML(df['yvel'],SM_WINDOW),
        # also get an x axis modifier, +1 to right, -1 to left
        x_dir = grouped_df['x'].transform(
            lambda x: (x.iloc[-1] - x.iloc[0])/np.absolute(x.iloc[-1] - x.iloc[0])
        )
    )
    # %% [markdown]
    # Identify start and end of window when fish crosses threshold
    # One way is to loop through all the epoch groups and every timepoint to find the start and end of each swim window. However, it takes ~8s to loop through one .dlm file in this way using the code below:
    # ```
    # start = time.time()
    # tmp = defaultdict(list)
    # for epochNum, group in grp_by_epoch(df)[['swimSpeed']]:
    #     for ind, spd in group.iterrows():
    #         tmp[epochNum].append(ind)
    # end = time.time()
    # ```
    # Therefore, an alternative method is used here in order to directly apply all calculations to the whole data frame:
    #
    # 1. assign a boolean column 'ifSwim'. True if swim speed excede the threshold
    # 2. convert boolean to int., True = 1, False = 0. Assign to a new column 'swimIndicator'
    # 3. to separate swim windoes between different epochs, set the start of each epoch to swimIndicator = 0
    # 4. apply diff().abs().cumsum() to 'swimIndicator', assign values to a new column 'locoIdx' (locomotion index). In 'locoIdx', a unique index is assigned for each window with a swim speed below or above the threshold. All windows that are identified as swim activities have odd indices and those with swim speed below the threshold have even indices
    # 5. filter for inter swim duration. Link 2 fast swim windows (odd locoIdx) that are closer than MIN_SWIM_INTERVAL by changing swimIndicators of the window between them (even locoIdx) from 0 to 1
    # 6. repeat step 4 to calculate adjusted locomotion indices

    # %%
    # Find swim windows

    speed_threshold = PROPULSION_THRESHOLD
    spd_window = df[['epochNum','absTime','swimSpeed','angVel']]
    # assign boolean to determine whether swim speed excede threshold
    spd_window = spd_window.assign(
        ifSwim = spd_window['swimSpeed']>= speed_threshold
    )
    # transfer boolean to int
    spd_window = spd_window.assign(
        swimIndicator = (spd_window['ifSwim']==True).astype(int)
    )
    # exclude bouts that start at the beginning of epochs by setting the beginning of each epoch to 0
    spd_window.loc[grp_by_epoch(spd_window)['swimIndicator'].transform('idxmin'), 'swimIndicator'] = 0
    # use cumsum to generate swim indices. All swims faster than threshold should have odd swim indices
    spd_window = spd_window.assign(
        locoIDX = spd_window['swimIndicator'].diff().abs().cumsum()
    )

    # %%
    # NOTE, MANUALLY CAPPING DETECTED FREQUENCY at 10HZ
    # check if SpdWindStarts are realistically delayed from each other to constitute separate bouts. If not, link these bouts.
    # get the first timepoint of each swim window
    fast_win_st = grp_by_swim(spd_window,'locoIDX')[['locoIDX', 'absTime']].head(1)
    fast_win_ed = grp_by_swim(spd_window,'locoIDX')[['locoIDX', 'absTime']].tail(1).reset_index(drop=True)
    # delay between fast windows = fast_win_st[i+1] - fast_win_ed[i], therefore, drop the first fast window start
    fast_win_st_shift = fast_win_st.iloc[1:].reset_index(drop=True)
    # initialize the locomotion index adjustment calculator as a dataframe
    loco_idx_adj_calculator = fast_win_ed[['locoIDX']]
    # calculate swim delays (bout intervals).
    loco_idx_adj_calculator = loco_idx_adj_calculator.assign(swimDelay = fast_win_st_shift['absTime'] - fast_win_ed['absTime'])
    #  In the matlab code, each speed window ends at the first frame when swim speed drops below threshold.
    #  Here, the fast_win_ed is the last frame above threshold.
    # loco_idx_adj_calculator['swimDelay'] = loco_idx_adj_calculator['swimDelay'] - timedelta(seconds=1/SAMPLE_RATE)
    # only keep bouts with delay > 0.1
    loco_idx_adj = loco_idx_adj_calculator.loc[loco_idx_adj_calculator['swimDelay'] < timedelta(seconds=MIN_SWIM_INTERVAL),'locoIDX']
    # get locomotion indices of the slow windows to adjust
    loco_idx_adj = loco_idx_adj + 1

    spd_window_adj = spd_window.copy()
    for window_idx in loco_idx_adj:
        spd_window_adj.loc[spd_window_adj.locoIDX==window_idx,'swimIndicator']=1

    spd_window_adj = spd_window_adj.assign(
        locoIDXadj = spd_window_adj['swimIndicator'].diff().abs().cumsum()
    )

    print(".", end = '')
    # %% [markdown]
    # ## Isolate bouts
    # Calculate duration from peak to trough of angular acceleration for each propulsion bout, in a window of 0.6s (+/- 12 samples) surounding

    # %%
    # First, get the theoretical bout windows based on the peak swim speed

    # grab the index of the rows with maximun swim speed within each swim window
    swim_spd_peak_idx = grp_by_swim(spd_window_adj,'locoIDXadj')['swimSpeed'].idxmax().swimSpeed  # added .swimSpeed for pandas 1.2 to drop the first col which is group keys
    # calculate the starts and ends of bout windows
    bout_window = pd.concat({
        'start':swim_spd_peak_idx-BOUT_WINDOW_HALF,
        'end':swim_spd_peak_idx+BOUT_WINDOW_HALF,
        'peak':swim_spd_peak_idx
    }, axis=1)

    # Then, check bout windows for every bouts within each epoch. Assign bout indices for grouping bouts

    # to avoid the new bout windows from crossing epochs, first, make an epoch generator
    grouped_epoch = iter(grp_by_epoch(spd_window_adj))
    # initialize a new dataframe
    spd_bout_window = pd.DataFrame()
    # get the first epoch
    current_epoch = pd.DataFrame(next(grouped_epoch)[1])
    # loop through bout_window, unpack start, end and peak for convenience
    for i, (start, end, peak) in bout_window.iterrows():
        # for each bout check if it is within the current epoch
        if peak > current_epoch.index[-1]:
            # if the peak of the bout is out of the current epoch, move to the next epoch
            current_epoch = pd.DataFrame(next(grouped_epoch)[1])
        # assign a bout index to the new bout window. index = i. start from 0
        # bout indices are only assigned to rows within the current epoch
        spd_bout_window = pd.concat([spd_bout_window,current_epoch.loc[start:end].assign(boutIDX = i)])
    spd_bout_window = spd_bout_window.reset_index(drop=False)

    # Note, at this point, spd_bout_window has duplicated rows assigned to adjacent bouts because the MIN_SWIM_INTERVAL is 100 ms but bout windows are 625 ms.
    # Nevertheless, the number of bouts remain the same
    if len(spd_bout_window.groupby('boutIDX').size()) == len(grp_by_swim(spd_window_adj,'locoIDXadj').size()):
        pass
    else:
        raise ValueError("The number of bouts windows doesn't match the number of speed windows.")

    # %%
    # Get swim window and bout window indices
    bout_idx = pd.DataFrame({'boutNum' : (list(range(len(swim_spd_peak_idx))))})
    bout_idx = bout_idx.assign(
        epochNum = df.loc[swim_spd_peak_idx,'epochNum'].values,
        peak_idx = swim_spd_peak_idx,
        swim_start_idx = grp_by_swim(spd_window_adj,'locoIDXadj').head(1).index,
        swim_end_idx = grp_by_swim(spd_window_adj,'locoIDXadj').tail(1).index,
        bout_start_idx = spd_bout_window.groupby('boutIDX', as_index=False, sort=False)['index'].head(1).values,
        bout_end_idx = spd_bout_window.groupby('boutIDX', as_index=False, sort=False)['index'].tail(1).values
    )

    # %%
    df = df.assign(
        swimWindow = spd_window_adj['locoIDXadj']
    )
    # initialize bout attributes
    bout_attributes = bout_idx.copy()

    # start and end coordinates of every bout window
    bout_start_loc = df.loc[bout_idx['bout_start_idx'], ['x','y']]
    bout_end_loc = df.loc[bout_idx['bout_end_idx'], ['x','y']]

    # start and end coordinates of every swim window
    swim_window_loc_diff = df.loc[bout_idx['swim_end_idx'],['x','y']].reset_index(drop=True) - df.loc[bout_idx['swim_start_idx'],['x','y']].reset_index(drop=True)

    # extract bout attributes
    bout_attributes = bout_attributes.assign(
        # peak swim speed in SWIM windows
        peakSpeed = df.loc[swim_spd_peak_idx,'swimSpeed'].values,
        # unsmoothed angVel.abs().max() in SWIM windows
        maxAbsRawAngVel = grp_by_swim(df, 'swimWindow')['angVel'].apply(
            lambda x: np.nanmax(np.absolute(x))).angVel,
        # unsmoothed peak angVel in SWIM windows
        peakRawAngVel = grp_by_swim(df, 'swimWindow')['angVel'].apply(
            lambda x: np.nanmax(np.absolute(x)) * (
                (np.nanmax(x)+np.nanmin(x)) / np.absolute(np.nanmax(x)+np.nanmin(x))
            )
        ).angVel,
        # bout displacement in BOUT windows
        propBoutDispl = np.linalg.norm(bout_start_loc.values - bout_end_loc.values, axis=1),
        # SWIM window Duration
        propBoutDur = grp_by_swim(df, 'swimWindow').size()['size']/SAMPLE_RATE,
        # cumulative bout heading of SWIM windows in degrees.
        # NOTE the swim windows here ends 1 frame earlier than that in the matlab code
        IEIheadings = np.degrees(
            np.arctan(((swim_window_loc_diff['y'])/(swim_window_loc_diff['x'].abs())).values)
        ),
        # time at peak swim speed
        peakTime = df.loc[swim_spd_peak_idx,'absTime'].values,
        # detect if bout is a 'mixed' event with positive and negative angVel over threshold
        boutMixedPeak = spd_bout_window.groupby('boutIDX')['angVel'].apply(lambda v: v.abs().max()),
        boutMixedIntegral = spd_bout_window.groupby('boutIDX')['angVel'].sum(),
        boutMixedValue = lambda x: x['boutMixedPeak']/x['boutMixedIntegral']
    )

    # %%
    # initialize more attributes
    bout_attributes = bout_attributes.assign(
        if_align = False,
        if_align_long = False,
        boutInflectAlign = np.nan,
        boutAccAlign = np.nan,
    )

    # decide which bouts to "align".
    # if window is far enough from epoch edge to allow alignment & spd during pre/post peak window are sufficiently low
    # YZ add code to get rid of bouts with only 0.025s above speed threshold
    for i, bout in bout_idx.iterrows():
        # if bouts meet criteria below, if_align = True
        if bout['peak_idx'] >= (df.loc[df['epochNum']== bout['epochNum']].index.min() + PRE_PEAK_FRAMES) and\
        df.loc[(bout['peak_idx']-frame_number250):bout['peak_idx'], 'swimSpeed'].min() < 3 and\
        df.loc[bout['peak_idx']:bout['bout_end_idx'], 'swimSpeed'].min() < 3: #  and\
        # bout['swim_end_idx']-bout['swim_start_idx'] > frame_number25: # added by YZ, get rid of bouts with only 1 frame above speed threshold
            # normal alignment, if bout peak far enough from epoch edges
            if bout['peak_idx'] <= (df.loc[df['epochNum']== bout['epochNum']].index.max() - POST_PEAK_FRAMES):
                bout_attributes.loc[i,'if_align'] = True
                # For inflection alignment, find the index of the frame with max speed inflection from boutWindowStart to boutWindowPeak
                bout_attributes.loc[i,'boutInflectAlign'] = df.loc[bout['bout_start_idx']:bout['peak_idx'],['swimSpeed']].diff().diff().idxmax().values
                # for alignment to max acceleration
                # in the Matlab code, since the smooth function doesn't actually smooth the first few values, this index is not accurate
                bout_attributes.loc[i,'boutAccAlign'] = smooth_series_ML(df.loc[bout['bout_start_idx']+frame_number250:bout['peak_idx'],'swimSpeed'], SM_WINDOW).diff().idxmax()

            # get bout number for longer duration alignment (20 extra frames for 40hz)
            if bout['peak_idx'] < (df.loc[df['epochNum']== bout['epochNum']].index.max() - BOUT_LONG_TAIL):
                bout_attributes.loc[i,'if_align_long'] = True

    # %% [markdown]
    # ## Extract values
    # Unlike the original Matlab code where different values are stored in their own variables, most multi-frame values extracted from all bouts will be stored in one multi-indexed dataframe (bout_res) for convenience. Other 1-value-per-bout data will be stored in another dataframe (bout_res2)
    #
    # Note: All the acceleration aligned results are skipped & All the bout pairs results are skipped
    #
    #     'propBoutAccAligned_speed'
    #     'propBoutAccAligned_pitch'
    #     PropBoutHeadingPairsFirst
    #     PropBoutHeadingPairsSecond
    #     PropBoutHeadingPairsTime
    #     PropBoutHeadingPairsFirstPitch
    #     PropBoutHeadingPairsFirstAccelerativeRotation
    #     PropBoutHeadingPairsSecondPitch
    #     PropBoutHeadingPairsSecondAccelerativeRotation
    #     PropBoutHeadingPairsSecondPitchRotation
    #     PropBoutHeadingPairsSecondPitchPreBout
    #     PropBoutHeadingPairsIEI
    #     PropBoutHeadingPairsIEIyVel
    #     PropBoutHeadingPairsSecondAlignedPitch
    #     PropBoutHeadingPairsSecondAlignedHeading
    # %%
    # now, all the information we need to extract bouts are stored in bout_attributes. Get bouts to align and re_index for upcoming analysis
    bout_aligned = bout_attributes.loc[bout_attributes['if_align']].reset_index(drop=True)

    if bout_aligned.empty:
        return "> no bout aligned > dlm file skipped" 
    # initialize result dataframe for bout alignment
    bouts = range(len(bout_aligned))
    frames = range(All_Aligned_FRAMES)
    index = pd.MultiIndex.from_product([bouts, frames], names=['bout_i', 'frame_i'])
    column = [  'propBoutAligned_angVel_hUp',
                'propBoutAligned_angVel_hDn',
                'propBoutAligned_speed_hUp',
                'propBoutAligned_speed_hDn',
                'propBoutAligned_pitch_hUp',
                'propBoutAligned_pitch_hDn',
                'propBoutAligned_instHeading',
                'propBoutAligned_angVel_flat',
                'propBoutAligned_speed_flat',
                'propBoutAligned_pitch_flat',
            ]
    bout_res = pd.DataFrame(index=index, columns=column)

    # set up idx for easy multi-indexing
    idx = pd.IndexSlice

    # initialize res2 dataframe for 1-per-bout numbers and assign some values
    bout_res2 = pd.DataFrame()
    bout_res2 = bout_res2.assign(
        aligned_time = df.loc[bout_aligned['peak_idx'], 'absTime'].values,
        aligned_time_hUp = np.datetime64('NaT'),
        aligned_time_hDn = np.datetime64('NaT'),
        # propBoutAligned_time    in the original code is the time in hours
        # propBoutAligned_trueTime    in the original code is time in 24hour day elapse
        propBoutAligned_dur = bout_aligned['propBoutDur'],     # = swim window duration
        propBoutAligned_displ = np.linalg.norm(
            df.loc[bout_aligned['peak_idx']+POST_PEAK_FRAMES, ['x','y']].reset_index(drop=True) \
            - df.loc[bout_aligned['peak_idx']-PRE_PEAK_FRAMES, ['x','y']].reset_index(drop=True) , \
            axis=1),
        propBout_initPitch = np.nan,
        propBout_initYPos = np.nan,
        propBout_deltaY = np.nan,
        propBout_netPitchChg = np.nan,
        propBout_matchIndex = bout_aligned['boutNum'],
        propBoutIEI_yDispl = np.nan,
        propBoutIEI_yDisplTimes = np.datetime64('NaT'),
        propBoutIEI_yDisplMatchedIEIs = np.nan,
        aligned_time_flat = np.datetime64('NaT'),
    )

    # some counters
    aligned_headUp_counter = 0
    aligned_headDn_counter = 0
    aligned_flat_counter = 0

    # same as before, set up a res2 dataframe for 1-value-per-bout data
    bout_long_res2 = pd.DataFrame(index=bout_aligned.loc[bout_aligned['if_align_long']].index)

    bout_matchIndex = bout_aligned.loc[bout_aligned['if_align_long'], 'boutNum']
    boutAlignLong = bout_aligned.loc[bout_aligned['if_align_long'], 'peak_idx']
    aligned_timeLong = df.loc[boutAlignLong.values, 'absTime']

    bout_long_res2 = bout_long_res2.assign(
        bout_matchIndex = bout_matchIndex.values,
        boutAlignLong = boutAlignLong.values,
        alignedLong_time = aligned_timeLong.values,
        propBoutLong_initPitch = np.nan,
        propBoutLong_initYPos = np.nan,
        propBoutLong_netPitchChg = np.nan,
    )

    # %%
    # align to each epoch
    bout_res_tmp1 = pd.concat([
        df.loc[bout['peak_idx']-PRE_PEAK_FRAMES:bout['peak_idx']+POST_PEAK_FRAMES,[  # select rows to concat
            'oriIndex',  # select columns to concat
            'absTime',  # added 06.17.2020
            'angVelSmoothed',
            'swimSpeed',
            'angAccel',
            'ang',
            'absy',
            'x',
            'y']
        ].assign(bout_i=[i]*All_Aligned_FRAMES, frame_i=range(All_Aligned_FRAMES))  # assign bout_i for each bout, assign frame_i for each row
        for i, bout in bout_aligned.iterrows()  # loop through bouts
    ]).set_index(['bout_i','frame_i']).rename(columns={  # reset index
            'oriIndex':'oriIndex',
            'absTime':'propBoutAligned_time',
            'angVelSmoothed':'propBoutAligned_angVel', # is smoothed!!!!!!!!!
            'swimSpeed':'propBoutAligned_speed',
            'angAccel':'propBoutAligned_accel',  # This is angAccel - calculated using unsmoothed angVel
            'ang':'propBoutAligned_pitch',
            'absy':'propBoutAligned_absy',
            'x':'propBoutAligned_x',
            'y':'propBoutAligned_y'
    })

    # align to inflection point of speaed (peak of 2nd derivative)
    bout_res_tmp2 = pd.concat([
        df.loc[bout['boutInflectAlign']-PRE_PEAK_FRAMES:bout['boutInflectAlign']+POST_PEAK_FRAMES,[
            'angVelSmoothed',
            'swimSpeed',
            'angAccel']
        ].assign(bout_i=[i]*All_Aligned_FRAMES, frame_i=range(All_Aligned_FRAMES))
        for i, bout in bout_aligned.iterrows()  # loop through bouts
        # add a condition for inflect alignment
        if bout['boutInflectAlign'] > PRE_PEAK_FRAMES and bout['boutInflectAlign'] < df.loc[df['epochNum']==bout['epochNum']].index.max()-POST_PEAK_FRAMES
    ]).set_index(['bout_i','frame_i']).rename(columns={
        'angVelSmoothed':'propBoutInflAligned_angVel',
        'swimSpeed':'propBoutInflAligned_speed',
        'angAccel':'propBoutInflAligned_accel'
    })

    bout_res = pd.concat([bout_res_tmp1,bout_res_tmp2],axis=1)

    # long bout tail alignment
    try:
        bout_long_res = pd.concat([
            df.loc[bout['peak_idx']-PRE_PEAK_FRAMES:bout['peak_idx']+BOUT_LONG_TAIL,[
                'angVelSmoothed',
                'swimSpeed',
                'angAccel',
                'ang']
            ].assign(bout_i=i, frame_i=range(BOUT_LONG_TAIL+PRE_PEAK_FRAMES+1))
            for i, bout in bout_aligned.iterrows()
            # add a condition for long bout tail alignment
            if bout['if_align_long']
        ]).set_index(['bout_i','frame_i']).rename(columns={
            'angVelSmoothed':'propBoutAlignedLong_angVel',
            'swimSpeed':'propBoutAlignedLong_speed',
            'angAccel':'propBoutAlignedLong_accel',
            'ang':'propBoutAlignedLong_pitch'
        })
    except:
        bout_long_res = pd.DataFrame()

    # %%
    # calculate heading values using (unchopped) df_chopped (Modified 2020.06.11)
    # for: each epoch
    #     if: x[-1] > x[1]  # >>
    #         then: move_angle = np.arctan2(yvel,xvel)
    #     if: x[-1] < x[1]  # <<
    #         then: flip x axis
    #               move_angle = np.arctan2(yvel,-xvel)

    bout_heading = pd.concat([
        df_chopped.loc[bout['peak_idx']-PRE_PEAK_FRAMES:bout['peak_idx']+POST_PEAK_FRAMES,[  # select rows to concat
            'xvel_sm',  # select columns to concat
            'yvel_sm',
            'x_dir',]
        ].assign(bout_i=[i]*All_Aligned_FRAMES, frame_i=range(All_Aligned_FRAMES))  # assign bout_i for each bout, assign frame_i for each row
        for i, bout in bout_aligned.iterrows()  # loop through bouts
    ]).set_index(['bout_i','frame_i'])

    # get the heading in -180:180 deg, which is the same unit/range as the original PropBoutAlignedHeading after U_D/R_L modifications
    # bout_res = bout_res.assign(
    #     propBoutAligned_instHeading = np.degrees(np.arctan2(
    #         bout_heading['yvel_sm'], (bout_heading['xvel_sm']*bout_heading['x_dir'])
    #     ))
    # )
    # YZ edited. instantaneous heading. replace the propBoutAligned_instHeading in previous versions
    # this heading notes the direction fish is moving, in a range -90:90 deg
    bout_res = bout_res.assign(
        propBoutAligned_instHeading = np.degrees(np.arctan2(
            bout_heading['yvel_sm'], np.absolute(bout_heading['xvel_sm'])
        ))
    )

    # %%
    # for rest of the values, let's do it bout by bout!
    for i, bout in bout_aligned.iterrows():
        aligned_peak = bout['peak_idx']
        aligned_start = bout['peak_idx'] - PRE_PEAK_FRAMES
        aligned_end = bout['peak_idx'] + POST_PEAK_FRAMES
        # # transfer values with multiple timepoints to bout_res
        # # to do this more efficiently, all the bout_res values are acquired with a single pd.concat. See the cell above.
        # bout_res.loc[idx[i,:], 'oriIndex'] = df.loc[aligned_start:aligned_end,'oriIndex'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_angVel'] = df.loc[aligned_start:aligned_end,'angVelSmoothed'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_speed'] = df.loc[aligned_start:aligned_end,'swimSpeed'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_accel'] = df.loc[aligned_start:aligned_end,'angAccel'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_pitch'] = df.loc[aligned_start:aligned_end,'ang'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_absy'] = df.loc[aligned_start:aligned_end,'absy'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_x'] = df.loc[aligned_start:aligned_end,'x'].values
        # bout_res.loc[idx[i,:], 'propBoutAligned_y'] = df.loc[aligned_start:aligned_end,'y'].values
        # put 1-per-bout values into res2
        bout_res2.loc[i, 'propBout_initPitch'] = df.loc[aligned_peak-frame_number250:aligned_peak-frame_number125, 'ang'].mean()
        bout_res2.loc[i, 'propBout_initYPos'] = df.loc[aligned_peak-frame_number250:aligned_peak-frame_number125, 'y'].mean()
        bout_res2.loc[i, 'propBout_deltaY'] = df.loc[aligned_peak+frame_number125:aligned_peak+frame_number250, 'y'].mean() - bout_res2.loc[i, 'propBout_initYPos']
        bout_res2.loc[i, 'propBout_netPitchChg'] = df.loc[aligned_peak+frame_number125:aligned_peak+frame_number250, 'ang'].mean() - bout_res2.loc[i, 'propBout_initPitch']


        # if the current (alignable) bout is not the last bout in the epoch
        if bout['boutNum'] < bout_attributes.loc[bout_attributes['epochNum']==bout['epochNum'],'boutNum'].max():
            swim_start = bout['swim_start_idx']
            swim_start_next = bout_attributes.loc[bout['boutNum']+1,'swim_start_idx']
            swim_end = bout['swim_end_idx'] + 1  # match swim_end to Matlab code by +1
            # and if IEI is long enough
            IEI_min = math.ceil(MIN_SWIM_INTERVAL * SAMPLE_RATE)  # 4 frames/100ms for 40Hz sample rate
            if (swim_start_next-1) - (swim_end+1) > IEI_min:
                # get IEI info
                bout_res2.loc[i, 'propBoutIEI_yDispl'] = df.loc[swim_start_next-1, 'y'] - df.loc[swim_end+IEI_min,'y']
                bout_res2.loc[i, 'propBoutIEI_yDisplTimes'] = bout_res2.loc[i, 'aligned_time']
                bout_res2.loc[i, 'propBoutIEI_yDisplMatchedIEIs'] = (swim_start_next - swim_start) / SAMPLE_RATE

        # separate by head up and head down
        if bout['peakRawAngVel'] > 0:
            aligned_headUp_counter += 1
            bout_res.loc[idx[i,:], 'propBoutAligned_angVel_hUp'] = bout_res.loc[idx[i,:], 'propBoutAligned_angVel']
            bout_res.loc[idx[i,:], 'propBoutAligned_speed_hUp'] = bout_res.loc[idx[i,:], 'propBoutAligned_speed']
            bout_res.loc[idx[i,:], 'propBoutAligned_pitch_hUp'] = bout_res.loc[idx[i,:], 'propBoutAligned_pitch']
            bout_res2.loc[i, 'aligned_time_hUp'] = bout_res2.loc[i,'aligned_time']
        else:
            aligned_headDn_counter += 1
            bout_res.loc[idx[i,:], 'propBoutAligned_angVel_hDn'] = bout_res.loc[idx[i,:], 'propBoutAligned_angVel']
            bout_res.loc[idx[i,:], 'propBoutAligned_speed_hDn'] = bout_res.loc[idx[i,:], 'propBoutAligned_speed']
            bout_res.loc[idx[i,:], 'propBoutAligned_pitch_hDn'] = bout_res.loc[idx[i,:], 'propBoutAligned_pitch']
            bout_res2.loc[i, 'aligned_time_hDn'] = bout_res2.loc[i,'aligned_time']

        # collect data for flat bouts: net rotation less than 3 deg
        if np.absolute(bout_res2.loc[i, 'propBout_netPitchChg']) <= 3:
            aligned_flat_counter += 1
            bout_res.loc[idx[i,:], 'propBoutAligned_angVel_flat'] = bout_res.loc[idx[i,:], 'propBoutAligned_angVel']
            bout_res.loc[idx[i,:], 'propBoutAligned_speed_flat'] = bout_res.loc[idx[i,:], 'propBoutAligned_speed']
            bout_res.loc[idx[i,:], 'propBoutAligned_pitch_flat'] = bout_res.loc[idx[i,:], 'propBoutAligned_pitch']
            bout_res2.loc[i, 'aligned_time_flat'] = bout_res2.loc[i,'aligned_time']

        # long bout tail alignment  - see the cell above

        if bout['if_align_long']:
            bout_long_res2.loc[i,'propBoutLong_initPitch'] = df.loc[aligned_peak-frame_number250:aligned_peak-frame_number125,'ang'].mean()
            bout_long_res2.loc[i,'propBoutLong_initYPos'] = df.loc[aligned_peak-frame_number250:aligned_peak-frame_number125,'y'].mean()
            bout_long_res2.loc[i,'propBoutLong_netPitchChg'] = df.loc[aligned_peak+frame_number125:aligned_peak+frame_number250,'ang'].mean() - bout_res2.loc[i, 'propBout_initPitch']
        #     bout_long_res.loc[idx[j,:], 'propBoutAlignedLong_angVel'] = df.loc[aligned_start:alignedLong_end,'angVelSmoothed'].values
        #     bout_long_res.loc[idx[j,:], 'propBoutAlignedLong_speed'] = df.loc[aligned_start:alignedLong_end,'swimSpeed'].values
        #     bout_long_res.loc[idx[j,:], 'propBoutAlignedLong_accel'] = df.loc[aligned_start:alignedLong_end,'angAccel'].values
        #     bout_long_res.loc[idx[j,:], 'propBoutAlignedLong_pitch'] = df.loc[aligned_start:alignedLong_end,'ang'].values
        #     j += 1
        # align for mixed bouts at two thresholds (skipped)

        # # align to inflection point of speaed (peak of 2nd derivative)  - see the cell above
        # if bout['boutInflectAlign'] > 30 and bout['boutInflectAlign'] < df.loc[df['epochNum']==bout['epochNum']].index.max() - 20:
        #     inflectAlign_start = bout['boutInflectAlign'] - 30
        #     inflectAlign_end = bout['boutInflectAlign'] + 20
        #     bout_res.loc[idx[i,:], 'propBoutInflAligned_angVel'] = df.loc[inflectAlign_start:inflectAlign_end,'angVelSmoothed'].values
        #     bout_res.loc[idx[i,:], 'propBoutInflAligned_speed'] = df.loc[inflectAlign_start:inflectAlign_end,'swimSpeed'].values
        #     bout_res.loc[idx[i,:], 'propBoutInflAligned_accel'] = df.loc[inflectAlign_start:inflectAlign_end,'angAccel'].values

    # %%
    # BE AWARE THAT SOME PROPERTIES ARE HARDED-CODED FOR 40HZ DATA. IF NOT 40HZ, ADJUST ACCORDINGLY
    # bout_res = bout_res.assign(
    #     propBoutAligned_instHeading = np.degrees(np.arctan2(bout_heading['xvel_sm'].values.astype(float), bout_heading['yvel_sm'].values.astype(float)))
    # )

    # x and y displacement for bout trajectory calculation
    # calculated as pre botu position - end bout position

    yy = (bout_res.loc[idx[:,bout_idx_peak+frame_number100],'propBoutAligned_y'].values - bout_res.loc[idx[:,bout_idx_peak-frame_number100],'propBoutAligned_y'].values).astype(float)
    absxx = np.absolute((bout_res.loc[idx[:,bout_idx_peak+frame_number100],'propBoutAligned_x'].values - bout_res.loc[idx[:,bout_idx_peak-frame_number100],'propBoutAligned_x'].values)).astype(float)

    # get more data
    # bout_res2, which will be saved as prop_bout2 in bout_data.h5, contains parameters that are one/bout
    bout_res2 = bout_res2.assign(
        epochBouts_indices = bout_aligned['peak_idx'],
        propBout_maxSpd = bout_aligned['peakSpeed'],  # modified 06.17.20
        epochBouts_heading = bout_res.loc[idx[:,bout_idx_peak],'propBoutAligned_instHeading'].values,
        epochBouts_preBoutPitch = bout_res.loc[idx[:,bout_idx_peak-frame_number100],'propBoutAligned_pitch'].values,
        # calculation of values below is done in plotting scripts, which gives more flexibility in trying different things
        # epochBouts_earlyRotations_28_30 = bout_res.loc[idx[:,29],'propBoutAligned_pitch'].values - bout_res.loc[idx[:,27],'propBoutAligned_pitch'].values,
        # epochBouts_earlyRotations = bout_res.loc[idx[:,30],'propBoutAligned_pitch'].values - bout_res.loc[idx[:,27],'propBoutAligned_pitch'].values,
        # epochBouts_lateRotations = bout_res.loc[idx[:,34],'propBoutAligned_pitch'].values - bout_res.loc[idx[:,31],'propBoutAligned_pitch'].values,
        epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
    )

    # %%
    # get/rename values from ALL bouts
    bout_attributes = bout_attributes.rename(columns={'peakTime':'propBout_time',
                                    'peakSpeed':'propBout_maxSpd',
                                    'maxAbsRawAngVel':'propBout_maxAngvel',
                                    'propBoutMixedIntegral':'propBoutMixed_integral',
                                    'boutMixedPeak':'propBoutMixed_peak',
                                    'boutMixedValue':'propBoutMixed_value',
                                    'peakRawAngVel':'propBout_peakAngvel',
                                    })
    # get fish length
    # bout_attributes = pd.merge(bout_attributes, fish_length, how='left', on='epochNum')
    bout_attributes = bout_attributes.assign(
        fisn_length = bout_attributes['epochNum'].map(fish_length.set_index('epochNum').to_dict()['fishLenEst'])
    )

    print(".", end="")
    # %% [markdown]
    # ## Extract IEI values
    #

    # %%
    # calculate corresponding pitch and angular velocity for IEIs. include window from SpdWindEnd to next SpdWindStart padded with 0.1s
    # differs from IEI detection and PropBout BoutWindows)
    # post_peak_buf = math.ceil(0.3 * SAMPLE_RATE)
    # pre_peak_buf = math.ceil(0.1 * SAMPLE_RATE)

    # Initialize IEI attributes
    IEI_attributes = bout_attributes[['boutNum','epochNum','swim_start_idx','swim_end_idx']]
    IEI_attributes = IEI_attributes.assign(
        # shift the end of each bout down by 1 row for easy iteration
        # NOTE: the swim_end indices here are 1 idx smaller than that in Matlab
        swim_end_shift = pd.concat([pd.DataFrame({'swim_end_idx':np.nan}, index=[0]), bout_attributes[['swim_end_idx']]]).reset_index(drop = True)
    )

    # Initialize res2
    IEI_res2 = bout_attributes[['boutNum','epochNum']]
    IEI_res2 = IEI_res2.assign(
        propBoutIEI = grp_by_epoch(bout_attributes)['swim_start_idx'].diff()/SAMPLE_RATE,
        propBoutIEItime = bout_attributes.loc[grp_by_epoch(bout_attributes).cumcount() >= 1, 'propBout_time'],
        # ignore PropBoutIEItrueTime
    )

    # drop first row of each epoch (rows with NA)
    rows_to_drop = list(IEI_res2.loc[IEI_res2['propBoutIEItime'].isna()].index)
    IEI_attributes.drop(rows_to_drop, inplace=True)
    IEI_res2.drop(rows_to_drop, inplace=True)
    
    if len(IEI_attributes) < 2:
        return "> no bout aligned > dlm file skipped"
    
    # all swim_end_shift and swim_start in the same row belong to the same epoch
    # reset index
    IEI_attributes = IEI_attributes.reset_index(drop=True)
    IEI_res2 = IEI_res2.reset_index(drop=True)
    # get some values
    IEI_res2 = IEI_res2.assign(
        # again, when using swim_end as an index, +1 to match the idx to Matlab
        propBoutIEI_pitchFirst = df.loc[IEI_attributes['swim_end_shift']+1+POST_BOUT_BUF, 'ang'].values,
        propBoutIEI_pitchLast = df.loc[IEI_attributes['swim_start_idx']-IEI_2_swim_buf, 'ang'].values,
        # use smoothed angVel for post bout vel
        propBoutIEI_angVel_postBout = df.loc[IEI_attributes['swim_end_shift']+1+POST_BOUT_BUF, 'angVelSmoothed'].values,
        propBoutIEI_angVel_preNextBout = df.loc[IEI_attributes['swim_start_idx']-IEI_2_swim_buf, 'angVelSmoothed'].values,
        # Other values to calculate
        propBoutIEI_pitch = np.nan,
        propBoutIEI_angVel = np.nan,
        propBoutIEI_angAcc = np.nan,
        propBoutIEI_pauseDur = np.nan,
        propBoutIEI_yvel = np.nan,
        IEI_matchIndex = np.nan,
        rowsInRes = np.nan,
        propBoutIEI_heading = np.nan,
    )

    # initialize res3 for timed IEI results (multi-indexed)
    IEI = range(len(IEI_attributes))
    frames = range(IEI_tail)
    index = pd.MultiIndex.from_product([IEI, frames], names=['IEI_i', 'frame_i'])
    column = [  'propBoutIEI_timedHeading',
                'propBoutIEI_timedPitch',
                'propBoutIEI_timedHeadingPre',
                'propBoutIEI_timedPitchPre',
            ]
    IEI_res3 = pd.DataFrame(index=index, columns=column, dtype=np.float64)

    # %%
    # extract data
    for i, row in IEI_attributes.iterrows():
        # get some index calculation done
        bout_end_post = row['swim_end_shift'] + 1 + POST_BOUT_BUF  # where last bout ends. to match the swim end in Matlab, +1
        bout_start_pre = row['swim_start_idx'] - IEI_2_swim_buf  # where next bout starts
        bout_end_5frames = row['swim_end_shift'] + 1 + math.ceil(0.05*SAMPLE_RATE)  # why use 0.05 but not POST_BOUT_BUF for duration calculation???????
        bout_start_4frames = row['swim_start_idx'] - PRE_BOUT_BUF
        # assign values. NOTE: smoothed results are used for angVel and angAccel
        IEI_res2.loc[i,'propBoutIEI_pitch'] = df.loc[bout_end_post:bout_start_pre,'ang'].mean(skipna=True)
        IEI_res2.loc[i,'propBoutIEI_angVel'] = df.loc[bout_end_post:bout_start_pre,'angVelSmoothed'].mean(skipna=True)
        IEI_res2.loc[i,'propBoutIEI_angAcc'] = df.loc[bout_end_post:bout_start_4frames,'angVelSmoothed'].diff().mean(skipna=True)
        IEI_res2.loc[i,'propBoutIEI_pauseDur'] = (bout_start_4frames - bout_end_5frames) / SAMPLE_RATE
        # why use 0.3 but not POST_BOUT_BUF for yvel???????????
        IEI_res2.loc[i,'propBoutIEI_yvel'] = df.loc[row['swim_end_shift']+1+math.ceil(0.3*SAMPLE_RATE):bout_start_4frames, 'yvel'].mean()
        IEI_res2.loc[i,'IEI_matchIndex'] = i
        IEI_res2.loc[i,'rowsInRes'] = bout_start_4frames - bout_end_5frames + 1
        # is IEI long enough?
        if row['swim_start_idx']-(row['swim_end_shift']+1) < IEI_tail:
            # if not
            IEI_res3.loc[idx[i,:],['propBoutIEI_timedHeading','propBoutIEI_timedPitch','propBoutIEI_timedHeadingPre','propBoutIEI_timedPitchPre']] = 500
        else:
            # if yes
            swim_end = row['swim_end_shift'] + 1
            swim_start = row['swim_start_idx']
            # NOTE: headings below are different from the Matlab code. X differences are not abs()
            # heading
            IEI_res2.loc[i,'propBoutIEI_heading'] = np.degrees(np.arctan(
                (df.loc[swim_start,'y'] - df.loc[swim_end,'y'])
                / np.absolute(df.loc[swim_start,'x'] - df.loc[swim_end,'x'])
                ))
            # timed heading and heading pre
            if df.loc[swim_end+IEI_tail, 'epochNum'] == row['epochNum']:
                # if there's enough rows in the current epoch for getting (swim_end + IEI_tail)
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedHeading'] = np.degrees(np.arctan2(
                    (df.loc[swim_end:swim_end+IEI_tail,'y'].diff().dropna().values),  # because of diff(), first value is na
                     (df.loc[swim_end:swim_end+IEI_tail,'x'].diff().abs().dropna().values)  # use abs() to get rid of x directionality
                ))
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedPitch'] = df.loc[swim_end:swim_end+IEI_tail-1,'ang'].values
            else:
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedHeading'] = 500
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedPitch'] = 500

            if df.loc[swim_start-IEI_tail, 'epochNum'] == row['epochNum']:
                # if there's enough rows in the current epoch for getting (swim_start - IEI_tail)
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedHeadingPre'] = np.degrees(np.arctan2(
                    (df.loc[swim_start-IEI_tail:swim_start,'y'].diff().dropna().values),
                     (df.loc[swim_start-IEI_tail:swim_start,'x'].diff().abs().dropna().values)
                ))
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedPitchPre'] = df.loc[swim_start-IEI_tail+1:swim_start,'ang'].values
            else:
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedHeadingPre'] = 500
                IEI_res3.loc[idx[i,:],'propBoutIEI_timedPitchPre'] = 500

    # for aligned values (multiple values for each IEI), use pd.concat (which is more efficient)
    IEI_res = pd.concat([
        df.loc[(
            IEI_attributes.loc[i,'swim_end_shift']+1+math.ceil(0.05*SAMPLE_RATE)  # bout_end_5frames
        ):(
            IEI_attributes.loc[i,'swim_start_idx']-PRE_BOUT_BUF  # bout_start_4frames
        ),['angVelSmoothed','ang','yvel']] for i in range(len(IEI_attributes)  # get these values for each IEI
    )], ignore_index=True)

    IEI_res = IEI_res.rename(columns = {'angVelSmoothed':'propBoutIEIAligned_angVel',
                                    'ang':'propBoutIEIAligned_pitch',
                                    'yvel':'propBoutIEIAligned_yvel'})

    # find matching IEI, pre-IEI pitch and post-IEI net rotation
    IEI_wolpert = pd.DataFrame({
        # wolpert values starts from the second IEI, thus exclude the first IEI match index
        'IEI_matchIndex': IEI_res2.loc[1:,'IEI_matchIndex'].tolist(),
        # only get the pre IEI value, exclude the last speed_window_start.diff(). Be aware that df.loc[a:b] includes both a and b
        'wolpert_IEI': IEI_res2.loc[:len(IEI_res2)-2,'propBoutIEI'].tolist(),
        # same as above, exclude the last pitch
        'wolpert_preIEI_pitch': IEI_res2.loc[:len(IEI_res2)-2,'propBoutIEI_pitchFirst'].tolist(),
        'wolpert_postIEI_netRot': (IEI_res2.loc[1:,'propBoutIEI_pitchFirst'].values - IEI_res2.loc[:len(IEI_res2)-2,'propBoutIEI_pitchLast'].values).tolist(),
        'wolpert_Time': IEI_res2.loc[1:,'propBoutIEItime'].tolist()
        })

    # %% [markdown]
    # ## Acqure epoch information

    # %%
    # epoch_attributes = pd.DataFrame(index=range(len(grp_by_epoch(df).size())))
    grouped_df = grp_by_epoch(df)
    # get angular vel at baseline speed. store separately. TO SAVE
    all_baseline_angVel = df.loc[df['swimSpeed']<BASELINE_THRESHOLD,['angVel','epochNum']]
    # get average of angvel per epoch
    epoch_attributes = pd.DataFrame(index=range(len(grouped_df.size())))
    epoch_attributes = epoch_attributes.assign(
        mean_bl_angVel = grp_by_epoch(all_baseline_angVel)[['angVel']].mean(),
        epoch_absTime = grouped_df.head(1)['absTime'].values,
        epoch_mean_angVel = grouped_df['angVel'].mean(),
        # why use velocity????? shouldn't be using swimSpeed??????
        # YZ 2020.5.27 change to swimSpeed
        epoch_pause_yvel = grouped_df.apply(lambda e: e.loc[e['swimSpeed']<BASELINE_THRESHOLD,'yvel'].mean()),
        epoch_bout_yvel = grouped_df.apply(lambda e: e.loc[e['swimSpeed']>PROPULSION_THRESHOLD,'yvel'].mean()),
        yvel_mean = grouped_df.apply(lambda e: e.loc[e['swimSpeed']>PROPULSION_THRESHOLD,'yvel'].mean()),
    ).reset_index()

    print(".", end='')

    # %% [markdown]
    # ## Grab all headings (movement angles)
    # ```
    # for: each epoch
    #     if: x[-1] > x[1]  # >>
    #         then: move_angle = np.arctan2(yvel,xvel)
    #     if: x[-1] < x[1]  # <<
    #         then: flip x axis
    #               move_angle = np.arctan2(yvel,-xvel)
    # ```

    # %%
    # below is for heading matched calculation

    # df_chopped dataframe has been created at the beginning of the function

    # chop top and bottom rows
    rows_to_drop = grouped_df.head(EDGE_CHOP).index.to_list() + grouped_df.tail(EDGE_CHOP).index.to_list()
    df_chopped.drop(axis=0, index=rows_to_drop, inplace=True)
    df_chopped = df_chopped.reset_index(drop=False)
    df_chopped.rename(columns={'index':'dfIdx'}, inplace=True)
    heading_res = df_chopped[['dfIdx','epochNum']]
    # now we can calculate move angles
    heading_res = heading_res.assign(
        moveAngle = np.degrees(np.arctan2(
            df_chopped['yvel_sm'], (df_chopped['xvel_sm']*df_chopped['x_dir'])
        )),
        # get other data that match up with headings
        headingMatched_ang = df_chopped['ang'],
        headingMatched_speed = df_chopped['swimSpeed'],
        headingMatched_angVel = df_chopped['angVel'],
        headingMatched_angAccel = df_chopped['angAccel'],
        headingMathced_time = df_chopped['absTime'],
        headingMatched_yvel = df_chopped['yvel_sm']
    )
    # calculate mean pitch and heading for epoch, and RMSpitch (line ~770 in Matlab)
    heading_res2 = grp_by_epoch(heading_res)[['headingMatched_ang','moveAngle']].mean()
    heading_res2 = heading_res2.rename(columns={'headingMatched_ang':'epochPitch','moveAngle':'epochHeading'})
    # calculate RMS, NOTE I don't understand why the RMS is calculated in the way below, but this is the code from Matlab
    heading_res2 = heading_res2.assign(
        # in deg/sec
        RMS_pitch = grp_by_epoch(df_chopped)['centeredAng'].apply(
            lambda x: np.sqrt(np.sum(x**2))/len(x)*SAMPLE_RATE
        ),
        # in mm/(sec^2)
        RMS_speed = grp_by_epoch(df_chopped)['velocity'].apply(
            lambda x: np.sqrt(np.sum(x**2))/len(x)*SAMPLE_RATE
        )
    )

    # output dictionary
    output = {'grabbed_all':df,  # note, this is different from MATLAB grabbed values (line 694). Here, only epochs with bouts (max_swimSpeed filtered) are included
              'baseline_angVel':all_baseline_angVel,  # angVel of swimSpeed < baseline threshold
              'bout_attributes':bout_attributes,
              'prop_bout_aligned':bout_res,
              'prop_bout2':bout_res2,
              'prop_bout_aligned_long':bout_long_res,
              'prop_bout_aligned_long2':bout_long_res2,
              'IEI_attributes':IEI_attributes,
              'prop_bout_IEI_aligned':IEI_res,
              'prop_bout_IEI2':IEI_res2,
              'prop_bout_IEI_timed':IEI_res3,
              'wolpert_IEI':IEI_wolpert,
              'epoch_attributes':epoch_attributes,
              'heading_matched':heading_res,
              'epoch_pitch_heading_RMS':heading_res2}
    aligned_bout_num = len(bout_res2)
    print(f" {aligned_bout_num} bouts aligned")

    return output

def run(filenames, folder, frame_rate):
    '''
    Loop through all .dlm, run analyze_dlm() and grab_fish_angle() functions
    Concatinate results from different .dlm files
    '''
    logger = log_vf_ana('vf_ana_log')
    logger.info(f'Folder analyzed: {folder}')
    logger.info(f"Program ver: {program_version}")

    # initialize output vars
    grabbed_all = pd.DataFrame()
    baseline_angVel = pd.DataFrame()
    bout_attributes = pd.DataFrame()
    prop_bout_aligned = pd.DataFrame()
    prop_bout2 = pd.DataFrame()
    prop_bout_aligned_long = pd.DataFrame()
    prop_bout_aligned_long2 = pd.DataFrame()
    IEI_attributes = pd.DataFrame()
    prop_bout_IEI_aligned = pd.DataFrame()
    prop_bout_IEI2 = pd.DataFrame()
    prop_bout_IEI_timed = pd.DataFrame()
    wolpert_IEI = pd.DataFrame()
    epoch_attributes = pd.DataFrame()
    heading_matched = pd.DataFrame()
    epoch_pitch_heading_RMS = pd.DataFrame()

    total_bouts_aligned = 0
    metadata_from_bouts = pd.DataFrame()

    # read ini files of dlm files, if there's any
    par_files = [name.split(".dlm")[0]+" parameters.ini" for name in filenames]
    all_ini_files = glob.glob(f"{folder}/*.ini")
    ini_files_to_read = list(set(par_files).intersection(all_ini_files))
    exp_parameters = pd.DataFrame()
    if ini_files_to_read:
        for i, this_par_file in enumerate(ini_files_to_read):
            this_par = read_parameters(this_par_file)
            this_par = this_par.assign(
                dlm_loc = filenames[i],
                dlm_size = os.path.getsize(filenames[i]),
                ini_loc = this_par_file,
                )
            exp_parameters = pd.concat([exp_parameters, this_par],ignore_index=True)
        exp_parameters = exp_parameters.sort_values(by=['filename']).reset_index(drop=True)
        exp_parameters.to_csv(f"{folder}/dlm metadata.csv")

    # analyze dlm
    for i, file in enumerate(filenames):
        logger.info(f"File {i}: {file[-19:]}")
        raw = read_dlm(i, file)
        analyzed, fish_length = analyze_dlm_resliced(raw, i, file, folder, frame_rate)
        if type(analyzed) == str:
            print(analyzed)
            logger.warning(analyzed)
            continue
        res = grab_fish_angle(analyzed, fish_length,frame_rate)
        if type(res) == str:
            print(res)
            logger.warning(res)
            continue 
        this_metadata = {
            'filename':os.path.basename(file)[0:15],
            'aligned_bout':len(res['prop_bout2']),
            'mean_fish_len':fish_length['fishLenEst'].mean(),
        }
        this_metadata = pd.DataFrame(data=this_metadata,index=[0])
        metadata_from_bouts = pd.concat([metadata_from_bouts,this_metadata])
        # transfer values to final var
        grabbed_all = pd.concat([grabbed_all, res['grabbed_all']], ignore_index=True)
        baseline_angVel = pd.concat([baseline_angVel, res['baseline_angVel']], ignore_index=True)
        bout_attributes = pd.concat([bout_attributes, res['bout_attributes']], ignore_index=True)
        prop_bout_aligned = pd.concat([prop_bout_aligned, res['prop_bout_aligned']], ignore_index=True)
        prop_bout2 = pd.concat([prop_bout2, res['prop_bout2']], ignore_index=True)
        prop_bout_aligned_long = pd.concat([prop_bout_aligned_long, res['prop_bout_aligned_long']], ignore_index=True)
        prop_bout_aligned_long2 = pd.concat([prop_bout_aligned_long2, res['prop_bout_aligned_long2']], ignore_index=True)
        IEI_attributes = pd.concat([IEI_attributes, res['IEI_attributes']], ignore_index=True)
        prop_bout_IEI_aligned = pd.concat([prop_bout_IEI_aligned, res['prop_bout_IEI_aligned']], ignore_index=True)
        prop_bout_IEI2 = pd.concat([prop_bout_IEI2, res['prop_bout_IEI2']], ignore_index=True)
        prop_bout_IEI_timed = pd.concat([prop_bout_IEI_timed, res['prop_bout_IEI_timed']], ignore_index=True)
        wolpert_IEI = pd.concat([wolpert_IEI, res['wolpert_IEI']], ignore_index=True)
        epoch_attributes = pd.concat([epoch_attributes, res['epoch_attributes']], ignore_index=True)
        heading_matched = pd.concat([heading_matched, res['heading_matched']], ignore_index=True)
        epoch_pitch_heading_RMS = pd.concat([epoch_pitch_heading_RMS, res['epoch_pitch_heading_RMS']], ignore_index=True)
        
        logger.info(f"Bouts aligned: {this_metadata.loc[0,'aligned_bout']}")
        

    # %%
    # concat metadata from bouts and metadata from ini. save in parent folder (condition folder)
    metadata_from_bouts.reset_index(drop=True, inplace=True)
    metadata_from_bouts = metadata_from_bouts.sort_values(by=['filename']).reset_index(drop=True)
    if_exp_metadata_arrScript = glob.glob(f"{folder}/*metadata.csv")
    if not ini_files_to_read:   # if no ini file detected
        if if_exp_metadata_arrScript:   # pull metadata from metadata.csv if possible
            if_exp_metadata_arrScript = if_exp_metadata_arrScript[0]
            exp_metadata_arrScript = pd.read_csv(if_exp_metadata_arrScript, index_col=0)

            exp_metadata_arrScript = exp_metadata_arrScript.sort_values(by=['filename']).reset_index(drop=True)
            metadata_merged = metadata_from_bouts.merge(exp_metadata_arrScript, on='filename')
        else:  # there's nothing. user must have manually transferred dlm files to this folder
            metadata_merged = pd.DataFrame()
            logger.warning("No metadata detected!")
    else:   # if metadata detected, exp_parameters should have values
        metadata_merged = metadata_from_bouts.merge(exp_parameters, on='filename')

    # get parent folder dir
    condition_folder = os.path.dirname(folder)
    exp_name = os.path.basename(folder)
    # get metadata csv name
    metadata_merged.to_csv(os.path.join(condition_folder,f"{exp_name} metadata.csv"))

    total_bouts_aligned = metadata_from_bouts['aligned_bout'].sum()
    # %%
    output_dir = folder
    grabbed_all.to_hdf(f'{output_dir}/all_data.h5', key='grabbed_all', mode='w', format='table')
    baseline_angVel.to_hdf(f'{output_dir}/all_data.h5', key='baseline_angVel', format='table')
    bout_attributes.to_hdf(f'{output_dir}/bout_data.h5', key='bout_attributes', mode='w', format='table')
    prop_bout_aligned.to_hdf(f'{output_dir}/bout_data.h5', key='prop_bout_aligned', format='table')
    prop_bout2.to_hdf(f'{output_dir}/bout_data.h5', key='prop_bout2', format='table')
    prop_bout_aligned_long.to_hdf(f'{output_dir}/bout_data.h5', key='prop_bout_aligned_long', format='table')
    prop_bout_aligned_long2.to_hdf(f'{output_dir}/bout_data.h5', key='prop_bout_aligned_long2', format='table')
    IEI_attributes.to_hdf(f'{output_dir}/IEI_data.h5', key='IEI_attributes', mode='w', format='table')
    prop_bout_IEI_aligned.to_hdf(f'{output_dir}/IEI_data.h5', key='prop_bout_IEI_aligned', format='table')
    prop_bout_IEI2.to_hdf(f'{output_dir}/IEI_data.h5', key='prop_bout_IEI2', format='table')
    prop_bout_IEI_timed.to_hdf(f'{output_dir}/IEI_data.h5', key='prop_bout_IEI_timed', format='table')
    wolpert_IEI.to_hdf(f'{output_dir}/IEI_data.h5', key='wolpert_IEI', format='table')
    epoch_attributes.to_hdf(f'{output_dir}/all_data.h5', key='epoch_attributes', format='table')
    heading_matched.to_hdf(f'{output_dir}/all_data.h5', key='heading_matched', format='table')
    epoch_pitch_heading_RMS.to_hdf(f'{output_dir}/all_data.h5', key='epoch_pitch_heading_RMS', format='table')

    # %%
    data_file_explained = pd.DataFrame.from_dict(
        {'all_data.h5':[
            'Contains following keys',
            'grabbed_all',
            'epoch_attributes',
            'baseline_angVel',
            'heading_matched',
            'epoch_pitch_heading_RMS',
            ],
         'grabbed_all':['Contains raw data from dlm files, excluding Epochs with no bout detected.'],
         'epoch_attributes':['Attributes for epochs'],
         'baseline_angVel':['all baseline angular velocity'],
         'heading_matched':['heading directions'],
         'epoch_pitch_heading_RMS':['mean pitch and heading for epoch, and RMSpitch'],

         'bout_data.h5':[
             'Bout data including following keys',
             'bout_attributes',
             'prop_bout_aligned',
             'prop_bout2',
             'prop_bout_aligned_long',
             'prop_bout_aligned_long2',
             ],
         'bout_attributes':['Basic bout attributes'],
         'prop_bout_aligned':['Aligned data are bouts aligned at the time of the peak speed including 500ms before and 300ms after.'],
         'prop_bout2':['one-per-bout parameters'],
         'prop_bout_aligned_long':['including bout data with 1s after the time of peak speed'],
         'prop_bout_aligned_long2':['attributes for long bouts'],

         'IEI_data.h5':[
         'Contains inter bout data. Includes following keys',
         'prop_bout_IEI_aligned',
         'prop_bout_IEI2',
         'prop_bout_IEI_timed',
         'wolpert_IEI',
         ],
         'prop_bout_IEI_aligned':['inter bout data, aligned'],
         'prop_bout_IEI2':['inter bout data'],
         'prop_bout_IEI_timed':[],
         'wolpert_IEI':[],
         }
    , orient='index')
    data_file_explained.to_csv(f'{output_dir}/data_file_explained.csv')

    catalog_all_data = pd.DataFrame.from_dict(
        {'grabbed_all':grabbed_all.columns.to_list(),
         'epoch_attributes':epoch_attributes.columns.to_list(),
         'baseline_angVel':baseline_angVel.columns.to_list(),
         'heading_matched':heading_matched.columns.to_list(),
         'epoch_pitch_heading_RMS':epoch_pitch_heading_RMS.columns.to_list(),
            }
    , orient='index')
    catalog_all_data.to_csv(f'{output_dir}/catalog all_data.csv')

    catalog_IEI_data = pd.DataFrame.from_dict(
        {
         'prop_bout_IEI_aligned':prop_bout_IEI_aligned.columns.to_list(),
         'prop_bout_IEI2':prop_bout_IEI2.columns.to_list(),
         'prop_bout_IEI_timed':prop_bout_IEI_timed.columns.to_list(),
         'wolpert_IEI':wolpert_IEI.columns.to_list() }
    , orient='index')
    catalog_IEI_data.to_csv(f'{output_dir}/catalog IEI_data.csv')

    catalog_bout_data = pd.DataFrame.from_dict(
        {
         'bout_attributes':bout_attributes.columns.to_list(),
         'prop_bout_aligned':prop_bout_aligned.columns.to_list(),
         'prop_bout2':prop_bout2.columns.to_list(),
         'prop_bout_aligned_long':prop_bout_aligned_long.columns.to_list(),
         'prop_bout_aligned_long2':prop_bout_aligned_long2.columns.to_list(),
         }
    , orient='index')
    catalog_bout_data.to_csv(f'{output_dir}/catalog bout_data.csv')
# %%
    info_dict = {
        'frame_rate':frame_rate,
        'fish_length':fish_length.mean()[1],
        'fish_length_std':fish_length.std()[1],
        'total_bouts_aligned':total_bouts_aligned,
        'ver':program_version
    }
    analysis_info = pd.Series(info_dict)
    analysis_info.to_csv(os.path.join(output_dir,'analysis info.csv'))
    