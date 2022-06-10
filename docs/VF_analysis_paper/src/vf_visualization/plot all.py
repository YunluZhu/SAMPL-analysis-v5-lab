from plot_IBIposture import plot_IBIposture
from plot_timeseries import (plot_raw, plot_aligned)
from plot_bout_frequency import plot_bout_frequency
from plot_kinetics import plot_kinetics
from plot_fin_body_coordination import plot_atk_ang_posture_chg
import matplotlib.pyplot as plt

def main(root):
    plt.close('all')
    plot_bout_frequency(root)
    
    plt.close('all')
    plot_IBIposture(root)
    
    plt.close('all')
    plot_kinetics(root)
    
    # Timeseries for aligned bouts may take long to plot for large dataset (>10GB)
    plt.close('all')
    plot_aligned(root)
    
    plt.close('all')
    plot_raw(root)
    
    plt.close('all')
    plot_atk_ang_posture_chg(root) 
    
if __name__ == "__main__":
    root_dir = input("- Which data to plot? \n")
    main(root_dir)