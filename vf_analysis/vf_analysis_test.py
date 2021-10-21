'''
20.06.10 VF analysis 
Take one path containing .dlm files
Call grab_fish_angle and analysis ALL .dlm files in the directory
Return values with ALL .dlm files analyzed
'''

import sys
import os,glob
from bout_analysis import grab_fish_angle_resliced
from bout_analysis import grab_fish_angle
import time

root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7DD_data_resliced/7dd_Lesion/190905"

def main(folder):
    filenames = glob.glob(f"{folder}/*.dlm") # subject to change
    print(f"\n{len(filenames)} file(s) loaded")
    grab_fish_angle_resliced.run(filenames, folder)
                
if __name__ == "__main__":
    main(root)

