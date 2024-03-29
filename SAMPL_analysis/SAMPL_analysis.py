'''
- What is this script
    This script is part of the SAMPL analysis pipeline.
    It analyzes free-swimming apparatus data (.dlm files), extracts swim bouts and save results as hdf5 files in the same folder
- How does it work
    User needs to specify the directory containing .dlm files (or subfolders with .dlm files) and the frame rate used to capture the data.
    This script runs the "grab_fish_angle" function which calls functions in the <preprocessing> & <bout_analysis> folders
- How to use it
    After running the script, user needs to follow instructions and specify data directory and frame rate using CLI
    Erros may occur if any of the .dlm files to be analyzed lacks a "alignable" swim bout. If this happens, please delete this .dlm file (which is usually unreasonably small) and rerun the script.
    An "aligned" swim bout contains data 500ms before and 300ms after the time of the peak speed.
    After running the script, pease refer to catalog.csv fiels for descriptions of the data extracted. A copy of catalog fiels can be found under <docs> folder.
- Requirments
    Please refer to the README file for required packages
'''

import sys
import os,glob
from bout_analysis import grab_fish_angle_v5
from bout_analysis.logger import log_SAMPL_ana

from tqdm import tqdm

def SAMPL_analysis(root,frame_rate, if_epoch_data=False):
    """Analyze behavior data. Extract bouts. Align bouts.

    Args:
        root (string): directory of behavior data to be analyzed. Data in all subfolders of the root directory will be analyzed. .dlm files in the same folder will be combined for bout extraction.
        frame_rate (int): Frame rate 
    """
    logger = log_SAMPL_ana('SAMPL_ana_log')
    logger.info(f"Analysis Started!")
    logger.info(f"Root dir: {root}")
    logger.info(f"Frame Rate: {frame_rate}")
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
            grab_fish_angle_v5.run(filenames, root, frame_rate, if_epoch_data)
            pbar.update(len(filenames)) # update progress bar after processing dlm in the current folder

        for path, dir_list, file_list in all_folders: # look for dlm in all subfolders
            # loop through each subfolder
            dir_list.sort()
            for folder_name in dir_list:
                # get the folder dir by joining path and subfolder name
                folder = os.path.join(path, folder_name)
                filenames = glob.glob(os.path.join(folder,"*.dlm"))
                if filenames:
                    print(f"\n\n- In {folder}")
                    grab_fish_angle_v5.run(filenames, folder, frame_rate, if_epoch_data)
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
    confirm = input("- Do you want to save epoch data? (y/n): ")
    if confirm == 'y':
        SAMPL_analysis(root_dir, frame_rate, if_epoch_data=True)
    elif confirm == 'n':
        SAMPL_analysis(root_dir, frame_rate, if_epoch_data=False)
    else:
        pass
    print("--- Analysis ended ---")
