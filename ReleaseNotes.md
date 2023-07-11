## Release notes

**v5.1.230711**

1. Bug fix in get_bout_features.py
2. New visualization scripts added.

**v5.1.230621**

1. Calculation of x, y, headx, heady improved. Solve the "Column not found" issue that may appear when calling columns repeatedly after groupby.
2. Compatibility checked. Analysis code should work with Pandas 1.5 and Python 3.10
3. Version control improved. analyze_dlm() and grab_fish_angle() gets their own version number which are saved in the analysis info file
4. Log total aligned bouts per folder

**v5.1.230131**

1. Visualization scripts cleaned up
2. New function: plt_categorical_grid() is used for all point plots with individual repeats shown in lines. Refer to *Visualization* section for use
3. New visualization plots
4. Speed up timeseries plotting by averaging traces beforehand

**v5.0.230111**

1. Speed threshold changed to 5 mm/s
2. Fixed a bug in analysis that randomly excludes ~half of the data
3. Increased max angular velocity threshold from 100 to 250
4. Saving all_data.h5 is now optional, see section "Analyze raw data files" for details

*Re-analyze your data is highly recommended*

**v4.1.220610**

1. Shortened aligned bout duration
2. Added arrangement scripts
3. Visualization code streamlined. Added functions to get features and kinetics
4. New Fin-Body algorithm
5. Visualization code now takes zeitgeber time (ztime) as a second input (other than `root`). Legal inputs are: `'day'`, `'night'`, and `'all'`.

**v4.3.220826**

1. Fixed a filter error on angular acceleration in analyze_dlm_v4.py. Now yields 50-80% more bouts
2. Added logging function. One log file will be generated the first time you run the ana code and will be updated every time you analyze a dataset.
3. Added the ability to skip dlm with no alignable bouts. With this version, to prevent error during ana process, just search under the root folder for “dlm”, delete any dlm file that are <1MB. Then you are safe to run the ana code.
4. Metadata handling updated. Can read ini files directly in the grab_fish_angle script and export metadata as a csv in parent folder
5. New visualization scripts.

**v4.4 221013**

1. Parameters redefined. See parameter table below for details
2. Fixed bug in `get_bout_features()` `get_bout_kinetics()` `get_IBIangles()` causing oversampling of bouts
