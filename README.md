# SAMPL Analysis and Visualization

This code is for analysis and visualization of data generated using free-swimming apparatus. 

The VF analysis package reads LabView .dlm files generated by free swimming vertical fish apparatus (preprocessing), extracts and analyzes bouts (bout_analysis), and makes figures (visualization).

Run `SAMPL_analysis/SAMPL_analysis_....py` to analyze .dlm files. Then, run individual visualization scripts under `SAMPL_visualization/` to make figures. See below for detailed instructions.

**v5.3.230816**

1. New analysis pipeline using Multiprocessing. Documentation to be updated.

**v5.2.230810**

1. `grab_fish_angle` updated to fix a bug during bout segmentation and generation of `spd_bout_window`. The bug is raised when an epoch contains no detectable bout. If you have not encountered `ValueError: The number of bouts windows doesn't match the number of speed windows.)` during analysis, there's no need to re-analyze the dataset.

**v5.2.230502**

1. New visualization plots (Navigation)
2. New functions in plot_functions

**v5.1.230131**

1. Visualization scripts cleaned up
2. New function: plt_categorical_grid() is used for all point plots with individual repeats shown in lines. Refer to *Visualization* section for use
3. New visualization plots for depth change, lift gain, and xy traces
4. Speed up timeseries plotting by averaging traces beforehand

**Known issues**

- analysis code can deal with .dlm with no bouts mostly. However, bout check has not been implemented to every single quality control filters, which means, though very unlikely, you may still get an error if there's no alignable bout. In another senario, if you have data that ends up giving 1 aligned bouts, analysis code will throw an error when it's trying to calculate the mean values.

- analysis won't go through if the last dlm file to be analyzed in a folder contains no alinable bouts

- analysis won't go through if a dlm file only contains 1 epoch

## Prerequisites and tips

Build with Python3.10. See `environment.yml` for a complete list of packages.
Below is a list of required packages:

- astropy=5.1
- pandas=1.4.4
- pytables=3.7.0
- matplotlib=3.5.2
- numpy=1.23.3
- scipy=1.9.1
- seaborn=0.12.0
- tqdm=4.64.1
- scikit-learn=1.1.1

1. Conda environment is recommended. Download miniconda here: <https://docs.conda.io/en/latest/miniconda.html>
2. Setting up conda envs can be the most time-consuming step. Be patient and prepare to Google a lot.
3. Visual Studio Code is a good IDE and is compatible with Jupyter Notebook
4. VS code Python Extension supports Interactive Window
    - You can create cells on a Python file by typing `# %%`
    - Use `Shift`+`Enter` to run a cell, the output will be shown in an interactive window

## Usage

### Contents

`docs` contains a copy of catalog files generated after running `.../SAMPL_analysis/SAMPL_analysis.py`

`docs/SAMPL_analysis_visualization_paper` contains complete code for the SAMPL method manuscript that should work out of the box.

`SAMPL_analysis` folder contains all the scripts for data analysis.

`SAMPL_visualization` includes all scripts for plotting.

`SAMPL_dataARR` contains code for raw data arrangement. These scripts read metadata files (.ini) and arrange raw data collected from multiple boxes into organized structure (see Data arrangement section below). Arrangement scripts are specific to the experiments so you may want to write your own code that works for your experimental conditions.

### Data arrangement

1. Organize .dlm files. Each folder with .dlm files will be recognized as one "experiment (exp)" during jackknife analysis. Therefore, if you want to combine all data from a certain clutch, put them into the same folder. See below for a sample structure. For the folders representing experimental conditions, 2 conditions separated by "_" are taken as inputs. e.g. `cond0_cond1`. For consistency, it is recommended to use `cond0` for the age of the fish and/or light-dark condition and mark the experimental condition using `cond1`.
2. However, for the analysis code to work, your data doesn't have to be in this structure. `SAMPL_analysis/SAMPL_analysis....py` looks for all .dlm under the directory and subfolders in the directory the user specifies. Therefore, it can be used to analyze data generated from a single experiment by giving it (in the example below) `root/7dd_ctrl/200607 ***` as the root directory. Again, all .dlm files under the same folder will be combined for analysis, if you want to treat them as different "conditions", move them into different parent folders and name the parent folders as described above.
3. It is recommended to write an arrangement code that reads metadata files (.ini) and organizes your .dlm data instead of moving files manually. See `SAMPL_dataARR` for some sample scripts.
4. For Jackknife resampling to work properly, make sure the exp folders under each conditions can be sorted in the same alphabetical order. In the example below, (if intended to compare against each other,) experiment folders in `7dd_ctrl` correspond with those under `7dd_condition`. Folder names for experiment folders under each conditions don't need to be the same but should be in the same alphabetical order. If multiple repeats (experiment folders) are generated in a single day, one may name the experiment folders as `exp1`, `exp2` etc.

```bash
├── root
    ├── 07dd_ctrl
    │   ├── 200607 ***
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   ├── 200611 ***
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    ├── 07dd_condition
    │   ├── 200607 ***
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   ├── 200611 ***
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    │   │   ├── ****.dlm
    └── 04dd_ctrl
        └── 20**** ***
            └── ****.dlm
```

### Analyze raw data files

To analyze data generated using the free-swimming apparatus:

1. Run `SAMPL_analysis/SAMPL_analysis....py`.
2. Check inputs to `SAMPL_analysis` to make sure `if_oil_fill_sb = False`. This parameter is for analyzing oil-filled swim bladder data, which disables the max angular acceleration filter.
2. Follow the instruction and input the root path that contains data files (.dlm) and corresponding metadata files (.ini). Determine whether to save all epochs that pass quality control, which *significantly increases file size*. Only Etimeseries_xytraces uses such data.
3. Follow the instruction and input the frame rate (in integer), and decide whether to save "epoch data that passes quality control" (y/n). If yes, epoch data will be saved to `all_data.h5`; if no, an empty `all_data.h5` will be generated. See notes for details.
4. The program will go through every data file in each subfolder (if there is any) and extract swim attributes.

When finished, there will be three hdf5 files (.h5) under each directory that contains data file(s) together with catalog files that explains the parameters extracted. A copy of catalog files can be found under `docs`.
All the extracted swim bouts under `bout_data.h5` are aligned at the time of the peak speed. Each aligned bout contains swim parameters from 500 ms before to 300 ms after the time of the peak speed.

**Notes** on data analysis

- All the .dlm data files under the same directory will be combined for bout extraction. To analyze data separately, please move data files (.dlm) and corresponding metadata files (.ini) into subfolders under the root path.
- Analysis program will stop if it fails to detect any swim bout in a data file (.dlm). To avoid this, please make sure all data files to be analyzed are reasonably large so that it contains at least one swim bout. Generally, we found > 10 MB being a good criteria.
- Please input the correct frame rate as this affects calculation of parameters. This program only accepts one frame rate number for each run. Therefore, all data files under the root path need to be acquired under the same frame rate.
- If saving epoch data is disabled, an EMPTY all_data.h5 will be saved (or will overwrite previously generated all_data.h5). This is designed to ensure all analyzed files are generated from the same ver. of the script. Saving epoch data significantly increases file size. It is recommended not to do so for most of the datasets and only re-analyze those that you need epoch data.

### Make figures

1. **IMPORTANT** update `SAMPL_visualization/plot_functions/get_data_dir.py` to specify the names of your datasets and the directory of it. `get_data_dir(pick_data)` is called by every *visualization script*. Therefore, instead of typing directories for different datasets to plot every time, specifying the name of your dataset in *visualization scripts* `pick_data = '<NAME OF YOUR DATASET>'` tells the script which data to plot. Also update the `get_figure_dir()` function in `get_data_dir.py`. This should be the root folder to save all plotted figures. Subfolders named by the name of your datadsets (input to `pick_data`) will be created under your `get_figure_dir(pick_data)` directory.
2. Run individual scripts under `SAMPL_visualization/`.
    - each visualization script takes a root directory including all analyzed data, which should be same as the one fed to the analysis code. Visualization scripts calls `get_data_dir.py` to look for data directories.
    - all visualization scripts get experimental conditions and age info from folder names
    - jackknife is used for resampling in some scripts

**Visualization scripts and function** explained

1. Plot bout timeseries data
    - `Btimeseries_1_bySpdUD.py` plot bout features as a function of time (time series). Bouts are segmented by peak swim speed & separated by pitch up vs down.
    - `Btimeseries_2_feature_corr.py` plot Pearson correlation coefficient of bout features at each time point against given parameter.
    - `Btimeseries_3_bySR.py` plot bout features as a function of time (time series). Bouts are segmented by signs of steering/righting rotation.
    - `Etimeseries_xytraces.py` plot x y position vs time of a single epoch containing one or multiple bouts
2. Plot bout features
    - `Bfeatures_1_features.py` plot individual bout features, segmented at pitch initial (at 10 deg) or by set point
    - `Bfeatures_2_features_std.py` plot standard deviation of individual bout features
    - `Bfeatures_3_distribution.py` looks at distribution of features. Also allows you to plot one feature against another in 2D histogram.
    - `Bfeatures_4_by....py` calculates binned average features by pitch or speed and plot the mean by pitch/speed bins.
    - `Bfeatures_5_globalCorr.py` plot correlation of every feature against each other.
3. Plot bout kinematics
    - `Bkinetics_1_parameters_bySpd.py` plot bout kinematic parameters and VS speed.
    - `Bkinetics_1_steering_righting_stats.py` focus on steering and righting
    - `Bkinetics_2_fin_body_coordination....py` plot fin-body coordination.
    - `Bkinetics_3_righting_scatter.py` scatter plot of righting (deceleration rot vs pitch initial)
    - `Bkinetics_5_steering_coefs.py` plot all coefs of steering fit
    - `Bkinetics_5_steeringRot_trajDev_coefs.py` plot trajectory deviation VS acceleration rotation
    - `Bkinetics_6_xyEfficacy.py` plot x/y efficacy, lift gain
4. Plot bout inter bout interval data
    - `IBI_1_pitch_mean.py` plot pitch distribution and its std() during inter bout interval (IBI).
    - `IBI_2_timing.py` plot bout frequency (reverse of IBI duration) as a function of pitch and fits it with a parabola.
5. Other plots
    - `stat_..._ROC.py` plot ROC curve for statistics. Working but in relative rough condition.
6. Navigation
    - `Navigation_1` looks at relation between a given feature of one bout and following bouts. Plots autocorrelation and auto-regression
    - `Navigation_2` looks at depth change during bouts and inter-bout intervals. Plots cumulative depth change of a series bouts as a function of posture of the first bout
    - `Navigation_3` plots standard deviation of bout features during series of bouts


### Parameters

| Parameters                | Unit | Definition                                                                                |
| ------------------------- | ---- | ----------------------------------------------------------------------------------------- |
| Pitch angle               | deg  | Angle of the fish on the pitch axis relative to horizontal                                 |
| Peak speed                | mm/s | Peak speed of swim bouts                                                                  |
| Initial pitch             | deg  | Pitch angle at 250 ms before the peak speed (-250 ms)                                     |
| Post-bout pitch           | deg  | Pitch angle at 100 ms after the peak speed (-100 ms)                                      |
| End pitch                 | deg  | Pitch angle at 200 ms after the peak speed (200 ms)                                       |
| Acceleration phase        |      | Before time of the peak speed                                                             |
| Deceleration phase        |      | After time of the peak speed                                                              |
| Total rotation            | deg  | Pitch change from initial (250 ms before) to end (200 ms after) time of the peak speed    |
| Bout trajectory           | deg  | Tangential angle of the trajectory at the time of the peak speed                          |
| Bout displacement         | mm   | Displacement of fish during a time window when speed is faster than 5mm/s                 |
| Inter-bout interval       | s    | Duration between two adjacent swim bouts                                                  |
| Inter-bout-interval pitch | deg  | Mean pitch angle during inter-bout interval                                               |
| Trajectory deviation      | deg  | Deviation of bout trajectory from initial pitch (250 ms before)                           |
| Steering rotation         | deg  | Change of pitch angle from initial (-250 ms) to the time of the peak speed                |
| Steering gain             |      | Slope of best fitted line of posture vs trajectory at the time of the peak speed          |
| Early rotation            | deg  | Change of pitch angle from initial to -40 ms (or time of maxAngvel)                       |
| Attack angle              | deg  | Deviation of bout trajectory from pitch at time of the peak speed                         |
| Fin-body ratio            |      | Maximal slope of best fitted sigmoid of attack angle vs early rotation                    |
| Righting rotation         | deg  | Change of pitch angle from time of the peak speed to post bout (100ms) or to end bout (200 ms) |
| Righting gain             |      | Numeric inversion of the slope of best fitted line of righting rotation vs initial pitch  |
| Set point                 | deg  | x intersect of best fitted line of righting rotation vs initial pitch                     |
| x/y efficacy              |      | Slope of best fitted line of x/y displ from preBout to postBout (-100 to 100 ms) VS peak pitch    |
| Depth change              | mm   | Displacement of a bout from pre to post in y axis (depth)       |
| Additional depth change   | mm   | Depth change minus predicted depth change due to propulsion based on peak pitch angle and x displ      |
| Lift gain                 |      | Slope of best fitted line of additional depth change VS depth change                     |

## Guides

### On analysis

1. Data analysis takes time. Since the script goes through all the subfolders under root directory, be smart with the root input. There's no need to re-analyze the dataset if the .dlm files haven't been changed. In another word, if you've added new .dlm files into an analyzed folder containing old .dlm files, make sure to re-analyze this folder.
2. Always carry the .ini metadata file when moving .dlm around or generate a metadata table containing experiment info for all the .dlm files.

### On plotting

1. Conditions (`cond0` `cond1`) are taken from parent folder names and sorted alphabetically. If you want them to be in a specific order, the easiest way to do is to add number before each condition, e.g.: `07dpf_1ctrl` `07dpf_2cond`.
2. Some scripts only compare data with different `cond1`, feel free to edit the scripts and make the comparison across other conditions.
3. Bouts are separated into "nose-up" and "nose-down" bouts based on their initial pitch. The cut point is at a fixed 10 deg for all the ages/conditions. This is generally true across all the dataset I've look at. An alternative way is to calculate the set point for each condition and split bouts by their set points. This can be easily done by `groupby(['conditions', 'age', 'repeats', 'whatever']).apply(get_kinetics())` or looping through every sub-condition to apply `pd.cut()`.

### On data interpretation

1. Take any results based on <1000 bouts with a grain of salt. Some parameters (such as bout timing parabola fit and righting gain) require a large number of bouts (>5000) to start to converge.  
2. Always look at the time series plots first. Strong phenotypes can be seen on averaged time series results.
3. Then, look at distributions of parameters.
4. `Inter-bout interval pitch` shows posture/stability of fish. The timing parabola fit tells you their "preferred" posture, baseline bout rate (which can also be seen in IEI distribution plots), and sensitivity to posture changes.
5. Kinetics tells you how fish coordinate propulsion and rotation in general. `fin_body` and `steering gain` demonstrate fin engagement.
6. Lastly, check bout features (`Bfeatures_features` and `Bfeatures_4_by...`). Any subtle differences in the way fish swims can be picked up here. However, Jackknife resampling may "exaggerate" differences across fish with different backgrounds, so pay attention to the y-axis range.

### Feeling Overwhelmed?

See `Feeling Overwhelmed?.md` for more tips on navigating SAMPL data analysis.