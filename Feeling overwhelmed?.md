# Navigate SAMPL data analysis
YZ 5-9-2023

## How to determine if I have enough bouts to draw a conclusion?

From our experience, any experiment with 4 or more *standard* repeats should yield enough data for pilot analysis.
One standard repeat is defined as a 48-hour dataset collected from 3 boxes per condition, with each box having 2-3 fish per narrow chamber or 5-8 fish per standard chamber.
This should give you more than 6000 bouts per condition, which is good enough to compare standard deviations of swim properties, such as `IBI pitch`, `peak pitch`, `trajectory`, `end pitch`, `end angVel`, etc.
6000 bouts should give you robust regression for Righting, Steering, and Timing. Fin-body ratio may require close to 10,000 bouts to converge.
Check the SAMPL manuscript for bout number required to resolve differences in kinematic parameters.

However, if the bout number per repeat is too low, variations of swim properties among fish from different clutches may become dominant and introduce inconsistency. Regression-based parameters may not converge easily, and even the standard deviation of parameters can vary quite a lot. In this case, segmenting/categorizing swim bouts and calculating the means may help. See the section below for details.

## I don't have enough bouts for regressions. Is there any parameter I can look at?

If bout number per condition/repeat is not ideal for regression analysis, one might look into the mean of basic bout properties. However, it is important to note that mean of bout properties works the best when bouts are properly categorized.
We recommende categorizing swim bouts into 2 or 4 categories by righting rotation and steering rotation.

1. Categorization by righting rotation or initial pitch

Fish adopt different strategies for nose-up and nose-down bouts. We found that the direction of righting rotation serves as a good indicator for nose-up/down bouts. We would recommend first segmenting all bouts into positive & negative righting rotations. Alternatively, you may separate bouts by pitch initial (-250ms) at 10deg or so, which usually yields similar results.

2. Categorization by righting and steering

By plotting angular velocity vs time, we found that bouts can be categorized into 4 types by the directions they rotate toward: Steering(+) Righting(+), Steering(–) Righting(–), Steering(+) Righting(–), and Steering(–) Righting(+).

Therefore, to further categorize bouts, we suggest separating by positive and netagive steering rotations.

3. Parameters worth checking

Below is a list of parameters that we suggest to check the mean of, for each bout categories:
- `trajectory`: direction of movement at the time of peak speed
- `peak posture`: posture at the time of peak speed
- `initial posture`: posture at -250ms
- `end posture`: posture at 200ms
- `initial angvel`: angular velocity at -250ms
- `end angvel`: angular velocity at 200ms
- `pitch change` or `total rotation`: posture change from -250 to 200ms
- `angvel change`: angular velocity change from -250 to 200ms
- `trajectory pre phased`: average direction of movement from -250ms to -100ms
- `righting rotation` and `steering rotation`

## The single fish dilemma

A very useful parameter is the standard deviation of pitch angles, which represents how well the fish are able to maintain their posture.
However, due to individual variations and slight differences between apparatuses, different fish from different boxes may yield similar STD but different MEANs.
Parameters of bouts pooled from multiple fish/boxes usually show a wider distribution than that of bouts collected from a single fish/single box. 
One way to overcome this is to concatenate bouts from multiple fish with similar levels of manipulation (# of cells activated/ablated, fluorescent levels etc.).
If you want to run individual fish and determine manipulation effects in a fish-by-fish base, we recommend collecting control data from the same fish before (or after) the treatment. 
Keep in mind that developmental effect is quite significant from day 4 to day 7 so we'd recommend comparing data from day 7 & 8.

## The notorious fin-body ratio

Fin-body ratio is the trickiest parameter to calculate. It requires fitting the data with a sigmoid and determining its max slope. Below are some tips of how to get it to work. 
We also discuss alternative methods to look at "coordination".

1. Exclude bouts slower than 7mm/s

Faster bouts show more significant correlation between attack angle and rotation. We exclude bouts that are slower than 7mm/s for high frame-rate datasets. For 40Hz data, since speed detection is less accurate, we keep all bouts > 5mm/s.

```Python
df_toplt = df_toplt.loc[df_toplt['spd_peak']>=7]
```

2. Exclude bouts that have nagetive attack angles AND greater-than-median acceleration rotations

This QC step significant facilitates the fit.

```Python
df_toplt = all_feature_cond.drop(all_feature_cond.loc[(all_feature_cond['atk_ang']<0) & (all_feature_cond['rot_full_accel']>all_feature_cond['rot_full_accel'].median())].index)
```

3. Determine what rotation to use

Ideally, we plot `attack angle` as a function of `rotation to max angvel`, because this accounts for time differences in pre-bout posture change for fish in different light conditions. In LIGHT, rotations start later and usually have shorting duration. In DARK, rotations start earlier. Therefore, we apply a series of sophisticated calculation to determine the time of max angvel for each dataset. If you do not want to calculate time of max angvel, -250ms to -40ms is a good general window for pre-bout rotation.

4. Alternatives to fin-body ratio

Essentially, fin-body ratio estimates the coordination between fin-based lift and body rotation. Alternatively, we can estimate the depth change generated by lift and compare that to the actual depth change to determine the contribution of fins. To do this: (see also `lift gain` calculation in `Bkinetics_6_xyEfficacy.py`)
- Calculate `depth change` of swim bouts. Depth change is defined as vertical displacement from -250ms to 200ms.
- Estimate propulsion direction. We assume that the direction of propulsion without fin contribution is the same as the angle of the fish (posture). You may use posture at time of peak speed for this, or calculate the mean pitch angle during acceleration (-250ms to 0).
- Determine displ in depth with no lift. To do this, multiply the absolute displacement in x axis from -250ms to 200ms by tangent(direction)
- Subtract empirical depth change by displ in depth with no lift. This gives you an estimation of the effect of lift on depth change. Let's call it `estimated lift`.
- Plot `estimated lift` as a function of `depth change`, fit with a line. The slope of the line is what we call `lift gain`.