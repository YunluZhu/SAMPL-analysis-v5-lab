import math
import os,glob
import pandas as pd

def get_frame_rate(dir):
    info_file = glob.glob(os.path.join(dir,'*analysis info.csv'))
    info = pd.read_csv(info_file[0], index_col=0, header=None).T
    frame_rate = int(info['frame_rate'])
    return frame_rate
    
def get_index(fr):
    peak_idx = math.ceil(0.5 * fr)
    total_aligned = math.ceil(0.5 * fr) + math.ceil(0.3 * fr) +1
    return peak_idx, total_aligned

