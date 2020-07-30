'''
20.06.10 VF analysis 
Take one path containing .dlm files
Call grab_fish_angle and analysis ALL .dlm files in the directory
Return values with ALL .dlm files analyzed
'''

import sys
import os,glob
from grab_fish_angle import grab_fish_angle
import time

def main(folder):
    filenames = glob.glob(f"{folder}/*.dlm") # subject to change
    print(f"\n{len(filenames)} file(s) loaded")
    grab_fish_angle.run(filenames, folder)
                
if __name__ == "__main__":
    root = "/Users/yunluzhu/Lab/! Lab2/Data/VF/vf_data/combined_TTau_data/7dd_TTau/200709"
    main(root)