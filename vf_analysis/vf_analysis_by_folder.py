'''
20.06.01 VF analysis cli - by folders
Take one command line input as the directory containing subfolders with .dlm files
Call grab_fish_angle and analysis .dlm files in the subfolders
Store analyzed values of .dlm files in subfolders within subfolders 
but not concatenate results across .dlm files in different subfolders
Note: The depth of subfolders containing .dlm files does not matter
'''

import sys
import os,glob
from bout_analysis import grab_fish_angle
import time

# if run this script directly, paste the root directory here
root = '/Users/yunluzhu/Lab/! Lab2/Data/VF/vf_data/combined_TTau_data/7dd_TSibs'

# callable function
def main(root):
    all_folders = os.walk(root)
    for path, dir_list, file_list in all_folders:
        # loop through each subfolder
        for folder_name in dir_list:
            # get the folder dir by joining path and subfolder name 
            folder = os.path.join(path, folder_name)        
            filenames = glob.glob(f"{folder}/*.dlm")
            # print(f"{len(filenames)} file(s) detected")
            if filenames:
                print(f"In {folder_name}")
                grab_fish_angle.run(filenames, folder)

# if run this script directly
if __name__ == "__main__":
    main(root)
