'''
220502 VF analysis with 
Take one command line input as the directory containing subfolders with .dlm files
Call grab_fish_angle and analysis .dlm files in the subfolders
Store analyzed values of .dlm files in subfolders within subfolders 
but not concatenate results across .dlm files in different subfolders
Note: The depth of subfolders containing .dlm files does not matter
'''

import sys
import os,glob
from bout_analysis import grab_fish_angle_v4
# import time
from tqdm import tqdm

def main(root,frame_rate):
    # for progress bar and time estimation (2022.0126 update)
    dlm_files_count = 0
    for _, _, files in os.walk(root):
        dlm_files_count = dlm_files_count + len([dlm_files for dlm_files in files if ".dlm" in dlm_files])
        
    all_folders = os.walk(root)
    with tqdm(total=dlm_files_count) as pbar:  # Do tqdm
         
        # determine if dlm is under root folder
        filenames = glob.glob(os.path.join(root,"*.dlm"))
        if filenames:  # if dlm under root, process them
            print(f"\n\n- In {root}")
            grab_fish_angle_v4.run(filenames, root, frame_rate)
            pbar.update(len(filenames)) # update progress bar after processing dlm in the current folder
            
        for path, dir_list, file_list in all_folders: # look for dlm in all subfolders
            # loop through each subfolder
            for folder_name in dir_list:
                # get the folder dir by joining path and subfolder name 
                folder = os.path.join(path, folder_name)        
                filenames = glob.glob(os.path.join(folder,"*.dlm"))
                if filenames:
                    print(f"\n\n- In {folder}")
                    grab_fish_angle_v4.run(filenames, folder,frame_rate)
                    pbar.update(len(filenames)) # update progress bar after processing dlm in the current folder
            

if __name__ == "__main__":
    # if want to use Command Line Inputs
    root_dir = input("- Where's the root folder? \n")
    frame_rate = input("- What's the frame rate in int.? \n")
    try:
        frame_rate = int(frame_rate)
    except ValueError:
        print("^ Not a valid number for frame rate!")
        sys.exit(1)
    print ("- Start to extract bouts from:", root_dir)
    confirm = input("- Proceed? (y/n): ")
    while confirm != 'n':
        if confirm == 'y':
            main(root_dir, frame_rate)
            break
        else:
            confirm = input("- Proceed? (y/n): ")
    print("--- Analysis ended ---")
