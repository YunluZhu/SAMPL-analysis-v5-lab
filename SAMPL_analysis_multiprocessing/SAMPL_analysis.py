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
import time

def SAMPL_analysis_mp(root,frame_rate, if_epoch_data=False, if_multiprocessing=True):
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
    # dlm_parent_folders = []
    dlm_directories = []
    dlm_input = []
    new_dlm_paths = []
    for parent_path, _, files in os.walk(root):
        files.sort()
        if len([dlm_files for dlm_files in files if ".dlm" in dlm_files]) > 0:
            new_dlm_paths = ([os.path.join(parent_path, dlm_files) for dlm_files in files if ".dlm" in dlm_files])
        if new_dlm_paths:
            # dlm_parent_folders.append(parent_path)
            dlm_directories.extend(new_dlm_paths)
            dlm_input.append((new_dlm_paths, parent_path, frame_rate, if_epoch_data))
        
    if if_multiprocessing and len(dlm_directories) > 5:
        grab_fish_angle_v5.runMP(dlm_input)

    else:
        with tqdm(total=len(dlm_input)) as pbar:  
            for filenames, root, frame_rate, if_epoch_data in dlm_input:
                # print(f"\n\n- In {root}")
                grab_fish_angle_v5.run(filenames, root, frame_rate, if_epoch_data)
                pbar.update(1)


if __name__ == "__main__":
    if_multiprocessing = True
    if_epoch_data = False
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
        if_epoch_data = True
        print("^ Saving raw epoch values.")
    if if_multiprocessing:
        print("^ Multiprocessing...")
    SAMPL_analysis_mp(root_dir, frame_rate, if_epoch_data=if_epoch_data, if_multiprocessing=if_multiprocessing)
    print("--- Analysis ended ---")
