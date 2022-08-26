# %%

import math
import numpy as np

# %%
fr = 40
FRAME_RATE = 40

peak_idx = math.ceil(0.5 * fr)
total_aligned = math.ceil(0.5 * fr) + math.ceil(0.3 * fr) +1

T_INITIAL = -0.25 #s
T_PREP_200 = -0.2
T_PREP_150 = -0.15
T_PRE_BOUT = -0.10 #s
T_POST_BOUT = 0.1 #s
T_post_150 = 0.15
T_END = 0.2
T_MID_ACCEL = -0.05
T_MID_DECEL = 0.05


idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
idx_mid_accel = int(peak_idx + T_MID_ACCEL * FRAME_RATE)
idx_mid_decel = int(peak_idx + T_MID_DECEL * FRAME_RATE)
idx_end = int(peak_idx + T_END * FRAME_RATE)

idx_prep_200 = int(peak_idx + T_PREP_200 * FRAME_RATE)
idx_prep_150 = int(peak_idx + T_PREP_150 * FRAME_RATE)
idx_post_150 = int(peak_idx + T_post_150 * FRAME_RATE)

idx_initial_phase = np.arange(idx_initial,idx_pre_bout)
idx_prep_phase = np.arange(idx_prep_200,idx_prep_150)
idx_accel_phase = np.arange(idx_pre_bout,peak_idx)
idx_decel_phase = np.arange(peak_idx,idx_post_bout)
idx_post_phase = np.arange(idx_post_150,idx_end)
# %%
T_start = -0.3
T_end = 0.25
idx_start = int(peak_idx + T_start * FRAME_RATE)
idx_end = int(peak_idx + T_end * FRAME_RATE)
idxRANGE = [idx_start,idx_end]
# %%
