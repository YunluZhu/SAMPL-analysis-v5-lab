'''
What:
    This script copies new data from the <root> folder to corresponding condition folders under the <organized> folder
How:
    Look for "<6-digit date> HC" folders under <root>
    Go through each of these folders (new exp):
        Find parameter files (.ini), read into dataframe
        Concat parameter dataframes
        Savs as "<date> metadata.csv" under "<date> HC"
        Create new folders named "<date>" under corresponding condition folders in the "HC_organized *" folder
        Filter rows in the metadata dataframe, categorize into corresponding conditions, for each condition:
            Save metadata to <date> folder under each condition
            Copy .dlm to <date> folder corresponding condition folders
What does it need from you:
    It asks for root directory containing "HC_organized *" (and "<6-digit date> HC", if there's new data to be organized)
    After detecting new data, it asks if you want to transfer the data or not
    If new data has been copied, please remember to move the original "<6-digit date> HC" folders into the "HC_archive" folder (or somewhere else)

NOTE
condition folders are hardcoded
'''

# %%
import os,glob
import configparser
import pandas as pd
import shutil
from tqdm import tqdm

# %%
def read_parameters(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    box_number = config.getint('User-defined parameters','Box number')
    genotype = config.get('User-defined parameters','Genotype').replace('"','')
    age = config.getint('User-defined parameters','Age')
    notes = config.get('User-defined parameters','Notes').replace('"','')
    initials = config.get('User-defined parameters','Inititals').replace('"','') # "Inititals" misspelled in .ini
    light_cycle = config.getint('User-defined parameters','Light cycle')
    dir = config.get('User-defined parameters','Save data to?').replace('"','')
    line_1 = config.getint('User-defined parameters','Mom line number')
    line_2 = config.getint('User-defined parameters','Dad line number')
    cross_id = config.get('User-defined parameters','cross ID').replace('"','')
    num_fish = config.getint('User-defined parameters','Num fish')
    filename = config.get('User-defined parameters','Filename').replace('"','')
    parameters = pd.DataFrame({
        'box_number':box_number,
        'genotype':genotype,
        'age':age,
        'notes':notes,
        'initials':initials,
        'light_cycle':light_cycle,
        'dir':dir,
        'line_1':line_1,
        'line_2':line_2,
        'cross_id':cross_id,
        'num_fish':num_fish,
        'filename':filename,
    }, index=[0])
    return parameters

# %%
def main(root):
    root_unquantified = glob.glob(f"{root}/* unquantified")
    root_quantified = glob.glob(f"{root}/* quantified")
    sibs = 'sibs'
    tau = 'tau'
    path_uq_sibs = os.path.join(root_unquantified[0],sibs)
    path_uq_tau = os.path.join(root_unquantified[0],tau)
    # path_q_sibs = os.path.join(root_quantified[0],sibs)
    # path_q_tau = os.path.join(root_quantified[0],tau)
    
    new_exp_folders = glob.glob(os.path.join(root_unquantified[0], ('[0-9]' * 6) + ' SF*'))

    if new_exp_folders:
        print("- New experiments found: ")
        for exp in new_exp_folders:
            print(exp)
    else:
        print("- No new experiment found")

    if new_exp_folders:        
        for exp in new_exp_folders:
            # get date
            exp_name = os.path.basename(exp)
            date = exp_name[0:6]
            
            all_exp_parameters = pd.DataFrame() # one row per dlm
            all_fish_matadata = pd.DataFrame() # one row per box
            all_folders = os.walk(exp) 
            # go through every box folder within one exp folder 
            for path, dir_list, file_list in all_folders:  
                for folder_name in dir_list:
                    old_folder = os.path.join(path, folder_name) 
                    if 'ctrl' in folder_name:
                        fish_id = int(date)*100
                    elif 'sib' in folder_name:
                        fish_id = int(date)*100
                    else:
                        fish_id = int(date)*100+int(folder_name) # unless is sibs
                    size = 0
                    for ele in os.scandir(old_folder):
                        size += os.path.getsize(ele)
                        
                    # get metadata
                    dlm_files = glob.glob(f"{old_folder}/*.dlm")
                    par_files = [name.split(".dlm")[0]+" parameters.ini" for name in dlm_files]
                    exp_parameters = pd.DataFrame()
                    for i, this_par_file in enumerate(par_files):
                        this_par = read_parameters(this_par_file)
                        this_par = this_par.assign(
                            dlm_loc = dlm_files[i],
                            ini_loc = this_par_file,
                            fish_id = fish_id,
                            dlm_size = os.path.getsize(dlm_files[i]),

                        )
                        if this_par.loc[0,'genotype'] != 'tau':
                            this_par['fish_id'] = int(date)*100  # sibs are numbered 00 
                        exp_parameters = pd.concat([exp_parameters, this_par],ignore_index=True)
                        
                    all_exp_parameters = pd.concat([all_exp_parameters, exp_parameters],ignore_index=True)
                    # modify this_box_parameters to combine parameters and get one row per box
                    this_fish_metadata = exp_parameters.loc[[0],['fish_id','box_number','genotype','light_cycle','num_fish']]
                    this_fish_metadata = this_fish_metadata.assign(
                        duration = len(set(exp_parameters['age'])),
                        file_size = size * 0.000001,  # convert to MB
                    )
                    all_fish_matadata = pd.concat([all_fish_matadata, this_fish_metadata],ignore_index=True)
                        
            # save parameter df
            all_exp_parameters.to_csv(os.path.join(exp,f"{date} dlm metadata.csv"))
            all_fish_matadata.to_csv(os.path.join(exp,f"{date} metadata_byFish.csv"))
            
        # %%
        # arrange files or not
        confirm = input("- Move files? (y/n): ")
        while confirm != 'n':
            if confirm == 'y':
                # copy and paste files
                for exp in new_exp_folders:
                    exp_name = os.path.basename(exp)
                    exp_metadata = pd.read_csv(f"{exp}/{date} dlm metadata.csv", index_col=0)
                    metadata_by_fish = pd.read_csv(f"{exp}/{date} metadata_byFish.csv", index_col=0)
                    # filter metadata for condition and save metadata
                    metadata_by_fish.sort_values(by='box_number',inplace=True)
                    metadata_by_fish = metadata_by_fish.reset_index(drop=True)
                    metadata_by_fish.to_csv(os.path.join(root_unquantified[0],f"{date} metadata_byFish.csv"))
                    # generate folders for each fish using fish_id
                    for index, fish in metadata_by_fish.iterrows():
                        if fish['genotype'] == 'tau':
                            path = path_uq_tau
                        else:
                            path = path_uq_sibs
                        this_fish_path = os.path.join(path,str(fish['fish_id']))
                        os.makedirs(this_fish_path, exist_ok=True)
                        # save dlm metadata for this fish
                        this_fish_dlm_metadata = exp_metadata.loc[exp_metadata['box_number']==fish['box_number']]
                        this_fish_dlm_metadata.to_csv(os.path.join(this_fish_path,f"{fish['fish_id']}dlm metadata.csv"))
                    
                    # move dlm files
                    for index, row in exp_metadata.iterrows():
                        if row['genotype'] == 'tau':
                            path = path_uq_tau
                        else:
                            path = path_uq_sibs
                        dest = os.path.join(path,str(row['fish_id']),f"{row['filename']}.dlm")
                        shutil.copyfile(row['dlm_loc'],dest)
                print("new data organized. please to archive ori data files!")
                break
            else:
                confirm = input("- Proceed? (y/n): ")

if __name__ == "__main__":
    dir_confirm = input("- Is this the correct root dir?\n  /Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish (y/n): ")
    if dir_confirm == 'y':
        root = "/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish"
    else:
        root = input("- Paste root dir here: ")
    main(root)