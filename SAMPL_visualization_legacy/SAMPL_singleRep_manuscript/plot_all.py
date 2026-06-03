"""
Plots all figures using data from one root directory. See individual function for details.
"""

from plot_IBIposture import plot_IBIposture
from plot_timeseries import (plot_raw, plot_aligned)
from plot_bout_timing import plot_bout_timing
from plot_kinematics import plot_kinematics
from plot_kinematics_jackknifed import plot_kinematics_jackknifed
from plot_fin_body_coordination import plot_fin_body_coordination
from plot_fin_body_coordination_byAngvelMax import plot_fin_body_coordination_byAngvelMax
from plot_parameters import plot_parameters
import matplotlib.pyplot as plt

def main(root, sample):
    plt.close('all')
    plot_bout_timing(root, sample_bout=sample)
    
    plt.close('all')
    plot_IBIposture(root, sample_bout=sample)
    
    plt.close('all')
    plot_parameters(root)
    
    plt.close('all')
    plot_kinematics(root, sample_bout=sample)

    # If to plot jackknifed kinematic parameters:
    # plt.close('all')
    # plot_kinematics_jackknifed(root, sample_bout=sample)
    
    # Timeseries for aligned bouts may take long to plot for large dataset (>10GB)
    plt.close('all')
    plot_aligned(root)
        
    # If to use fixed time of max angvel to calculate steering related rotatioin (-250 to -40 ms)
    # plt.close('all')
    # plot_fin_body_coordination(root, sample_bout=sample)
        
    plt.close('all')
    plot_fin_body_coordination_byAngvelMax(root, sample_bout=sample)
    
    plt.close('all')
    plot_raw(root)
    
if __name__ == "__main__":
    root_dir = input("- Which data to plot? \n")
    sample = input("- How many bouts to sample from each repeat? ('0' for no sampling): ")
    main(root_dir, sample)