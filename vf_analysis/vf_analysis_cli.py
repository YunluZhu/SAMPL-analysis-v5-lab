'''
20.06.01 VF analysis cli
Take one command line input as the directory containing .dlm files
Call grab_fish_angle and analysis ALL .dlm files in the directory
Return values with ALL .dlm files analyzed
'''

import sys
import os,glob
from grab_fish_angle import grab_fish_angle
import time

def main(folder):
    filenames = glob.glob(f"{folder}/*.dlm") # subject to change
    print(f"{len(filenames)} file(s) loaded")
    grab_fish_angle.run(filenames, folder)
                
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        # main(args[0])
        main("/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/test_data/4dd_Sibs/200302 DD 4d neg num3")
    else:
        print('too many/few args, should just be the file path!')
        