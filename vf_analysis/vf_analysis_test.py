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
    main("/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/200206 DD 7dpf NTau neg num6")