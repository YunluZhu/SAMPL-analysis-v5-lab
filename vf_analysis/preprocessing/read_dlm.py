'''
Read one .dlm file and return a dataframe
Modified from:
analyzeFreeVerticalGrouped2.m by DEE 1.30.2015
    "the LabView code returns a value to mark an "epoch," which is a continuous series of frames that had at least one identified particle 
+ lines to output head location in addition to body, for detection direction of movement." 
'''

import pandas as pd

def read_dlm(i, filename):
    # read_dlm takes file index: i, and the file name end with .dlm
    # file_path = os.getcwd() # need to be defined/entered when calling as a function or use command line input
    # filenames = os.listdir
    # filenames = glob.glob(f"{file_path}/*.dlm") # subject to change 
    col_names = ['time','fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']
    try:
        raw = pd.read_csv(filename, sep="\t",names = col_names) # load .dlm
    except:
        print(f"No .dlm file found in the directory entered")
    else:
        print(f"File {i+1}: {filename[-19:]}", end=' ')
        
    # Clear original time data stored in the first row
    raw.loc[0,'time'] = 0
    # data error results in NA values in epochNum, exclude rows with NA
    raw.dropna(inplace=True)
    # rows with epochNum == NA may have non-numeric data recorded. In this case, change column types to float for calculation. not necessary for most .dlm.
    raw[['fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']] = raw[['fishNum','ang','absx','absy','absHeadx','absHeady','col7','epochNum','fishLen']].astype('float64',copy=False)
    return raw
  