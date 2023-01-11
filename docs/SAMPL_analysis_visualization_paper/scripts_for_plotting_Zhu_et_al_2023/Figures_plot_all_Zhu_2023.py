from Fig2_parameter_distribution_prob import Fig2_parameter_distribution_prob
from Fig2_parameters_alignedAverage import Fig2_parameters_alignedAverage
from Fig2_throughput import Fig2_throughput
from Fig2_epoch_timeseries import Fig2_single_epoch
from Fig2_fly_worm import Fig2_fly_worm_epoch
from Fig3_bout_timing import Fig3_bout_timing
from Fig4_steering_fit import Fig4_steering_fit
from Fig4_trajDeviation_pitchChg import Fig4_trajDeviation_pitchChg
from Fig5_fin_body_coordination import Fig5_fin_body_coordination
from Fig5_time_of_maxAngvel import Fig5_time_of_maxAngvel
from Fig6_pitch_timeseries import Fig6_pitch_timeseries
from Fig6_righting_fit import Fig6_righting_fit
from Fig7_bkg_IBI import Fig7_bkg_IBI
from Fig7_bkg_timing import Fig7_bkg_timing
from Fig7_bkg_steering_righting import Fig7_bkg_steering_righting
from Fig7_bout_number import Fig7_bout_number
from Fig7_bkg_fin_body_coordination import Fig7_bkg_fin_body
from Fig8_CI_width import Fig8_CI_width
from Fig8_sims_effectSize import Fig8_sims_effectSize 
import matplotlib.pyplot as plt
import os

def data_dir():
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root_7dd_finless = os.path.join(data_dir, "DD_finless")
    root_7dd = os.path.join(data_dir, "DD_7dpf")
    root_singleEpoch = os.path.join(data_dir, "movie_data_singleEpoch")
    fly_worm_epoch = os.path.join(data_dir, "fly worm data")
    data_directory_dict = {
        'DD_7dpf': root_7dd, # directory for folder "DD_7dpf"
        'finless': root_7dd_finless, # directory for folder "DD_finless"
        'epoch': root_singleEpoch,
        'fly_worm': fly_worm_epoch,
    }
    return data_directory_dict

root_dict = data_dir()

# ––– Figure 2: swim parameter distribution and timeseries

# Fig2_parameter_distribution_prob(root_dict['DD_7dpf'])
# plt.close('all')
# Fig2_parameters_alignedAverage(root_dict['DD_7dpf'])
# plt.close('all')
# Fig2_throughput(root_dict['DD_7dpf'])
# plt.close('all')
# Fig2_single_epoch(root_dict['epoch'])
# plt.close('all')
Fig2_fly_worm_epoch(root_dict['fly_worm'])
# plt.close('all')

# # ––– Figure 3: bout timing and sensitivity

# Fig3_bout_timing(root_dict['DD_7dpf'])
# plt.close('all')

# # ––– Figure 4: steering gain

# Fig4_trajDeviation_pitchChg(root_dict['DD_7dpf'])
# plt.close('all')
# Fig4_steering_fit(root_dict['DD_7dpf'])
# plt.close('all')

# # ––– Figure 5: fin-body coordination

# Fig5_time_of_maxAngvel(root_dict['DD_7dpf'])
# plt.close('all')
# Fig5_fin_body_coordination(root_dict['DD_7dpf'], root_dict['finless'])
# plt.close('all')

# # ––– Figure 6: righting gain

# Fig6_pitch_timeseries(root_dict['DD_7dpf'])
# plt.close('all')
# Fig6_righting_fit(root_dict['DD_7dpf'])
# plt.close('all')

# # ––– Figure 7: measurements of different zebrafish strains

# Fig7_bkg_IBI(root_dict['DD_7dpf'])
# plt.close('all')
# Fig7_bkg_timing(root_dict['DD_7dpf'])
# plt.close('all')
# Fig7_bkg_steering_righting(root_dict['DD_7dpf'])
# plt.close('all')
# Fig7_bout_number(root_dict['DD_7dpf'])
# plt.close('all')
# Fig7_bkg_fin_body(root_dict['DD_7dpf'])
# plt.close('all')

# # ––– Figure 8: statistics of kinetic regression
# # NOTE slow to plot

# Fig8_CI_width(root_dict['DD_7dpf'])
# plt.close('all')
# Fig8_sims_effectSize(root_dict['DD_7dpf'])
# plt.close('all')


    


