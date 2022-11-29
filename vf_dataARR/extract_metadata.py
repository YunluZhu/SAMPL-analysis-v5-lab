'''
What:
    This script copies new data from the <root> folder to corresponding condition folders under the <organized> folder
How:
    Look for "<6-digit date> r?" folders under <root>
    Go through each of these folders (new exp):
        Find parameter files (.ini), read into dataframe
        Concat parameter dataframes
        Savs as "<date> dlm metadata.csv" under "<date> r?"
        Create new folders named "<r?>" under corresponding condition folders in the "long_organized *" folder
        Filter rows in the metadata dataframe, categorize into corresponding conditions, for each condition:
            Save metadata to <date> folder under each condition
            Copy .dlm to <date> folder corresponding condition folders
What does it need from you:
    It asks for root directory containing "long_organized *" (and "<6-digit date> r?", if there's new data to be organized)
    After detecting new data, it asks if you want to transfer the data or not
    If new data has been copied, please remember to move the original "<6-digit date> r?" folders into the "long_archived" folder (or somewhere else)

NOTE
'''

# %%
import os,glob
import configparser
import pandas as pd
import shutil
# def main(root,frame_rate):
    # for progress bar and time estimation (2022.0126 update)
# %%
def read_parameters(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    box_number = config.getround_half_up('User-defined parameters','Box number')
    genotype = config.get('User-defined parameters','Genotype').replace('"','')
    age = config.getround_half_up('User-defined parameters','Age')
    notes = config.get('User-defined parameters','Notes').replace('"','')
    initials = config.get('User-defined parameters','Inititals').replace('"','')
    light_cycle = config.getround_half_up('User-defined parameters','Light cycle')
    dir = config.get('User-defined parameters','Save data to?').replace('"','')
    line_1 = config.getround_half_up('User-defined parameters','Mom line number')
    line_2 = config.getround_half_up('User-defined parameters','Dad line number')
    cross_id = config.get('User-defined parameters','cross ID').replace('"','')
    num_fish = config.getround_half_up('User-defined parameters','Num fish')
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

def get_cond_data(df,cond_keys):
    '''
    filter df and get rows matching cond
    '''
    if cond_keys == '4s':
        output = df.loc[(df['age']<6) & (df['genotype']=='sibs')]
    elif cond_keys == '4t':
        output = df.loc[(df['age']<6) & (df['genotype']=='tau')]
    elif cond_keys == '7s':
        output = df.loc[(df['age']>6) & (df['age']<9) & (df['genotype']=='sibs')]
    elif cond_keys == '7t':
        output = df.loc[(df['age']>6) & (df['age']<9) & (df['genotype']=='tau')]
    elif cond_keys == '14s':
        output = df.loc[(df['age']>13) & (df['age']<16) & (df['genotype']=='sibs')]
    elif cond_keys == '14t':
        output = df.loc[(df['age']>13) & (df['age']<16) & (df['genotype']=='tau')]
    return output

# %%
def main(root):
    CONDITIONS = {
        '4s':'04_1sibs',
        '4t':'04_2tau',
        '7s':'07_1sibs',
        '7t':'07_2tau',
        '14s':'14_1sibs',
        '14t':'14_2tau',
    }
    # root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Longitudinal'
    root_organized = glob.glob(f"{root}/*organized*")
    new_exp_folders_are_like = os.path.join(root, ('[0-9]' * 6) + ' r*')
    new_exp_folders = glob.glob(new_exp_folders_are_like)

    path4conditions = {}
    for i in CONDITIONS.keys():
        path4conditions[i] = os.path.join(root_organized[0],CONDITIONS[i])

    if new_exp_folders:
        print("- New experiments found: ")
        for exp in new_exp_folders:
            print(exp)
    else:
        print("- No new experiment found")

    # %%
    if new_exp_folders:        
        for exp in new_exp_folders:
            # get date
            exp_name = os.path.basename(exp)
            date = exp_name[0:6]
            repeat_num = exp_name[8:]
            # concatenate all parameter files from one experiment, then arrange data
            exp_parameters = pd.DataFrame()
            all_folders = os.walk(exp)
            for path, dir_list, file_list in all_folders:  
                for folder_name in dir_list:
                    folder = os.path.join(path, folder_name)        
                    dlm_files = glob.glob(f"{folder}/*.dlm")
                    par_files = [name.split(".dlm")[0]+" parameters.ini" for name in dlm_files]
                    for i, this_par_file in enumerate(par_files):
                        this_par = read_parameters(this_par_file)
                        this_par = this_par.assign(
                            dlm_loc = dlm_files[i],
                            ini_loc = this_par_file,
                            dlm_size = os.path.getsize(dlm_files[i]),
                            repeat_num = repeat_num,
                        )
                        exp_parameters = pd.concat([exp_parameters, this_par],ignore_index=True)
            # save parameter df
            exp_parameters.to_csv(f"{exp}/{date} dlm metadata.csv")

        # %%
        # arrange files or not
        confirm = input("- Move files? (y/n): ")
        while confirm != 'n':
            if confirm == 'y':
                # copy and paste files
                for exp in new_exp_folders:
                    exp_name = os.path.basename(exp)
                    date = exp_name[0:6]
                    repeat_num = exp_name[8:]
                    # first, make date folders under each condition
                    this_path4conditions = {}
                    for i in CONDITIONS.keys():
                        this_path4conditions[i] = os.path.join(path4conditions[i],f"r{repeat_num}")
                        os.makedirs(this_path4conditions[i], exist_ok=True)
                    
                    exp_metadata = pd.read_csv(f"{exp}/{date} dlm metadata.csv", index_col=0)
                                    
                    # filter metadata for each condition
                    for i in CONDITIONS.keys():
                        # loop through conditions, get metadata filtered
                        this_cond_metadata = get_cond_data(exp_metadata,i)
                        dest_dir = this_path4conditions[i]
                        for index, row in this_cond_metadata.iterrows():
                            # for the files matching current condition, move dlm
                            tmp_dest_file = os.path.join(dest_dir,f"{row['filename']}.dlm")
                            shutil.copyfile(row['dlm_loc'],tmp_dest_file)
                            # save metadata
                            tmp_dest_file = os.path.join(dest_dir,f"r{repeat_num} metadata.csv")
                            this_cond_metadata.to_csv(tmp_dest_file)
                        
                print("new data organized. please archive ori data files!")
                break
            else:
                confirm = input("- Proceed? (y/n): ")

if __name__ == "__main__":
    root = input("- Paste root dir here: ")
    main(root)