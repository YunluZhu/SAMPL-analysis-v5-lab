# Navigate SAMPL data analysis
YZ 5-9-2023

## How do I know if I have enough bouts to draw a conclusion?

From our experience, any experiment with 4 or more *standard* repeats should yield enough data for pilot analysis.
One standard repeat is defined as a 48-hour dataset collected from 3 boxes per condition, with each box having 2-3 fish per narrow chamber or 5-8 fish per standard chamber.
This should give you more than 6000 bouts per condition, which is good enough to compare standard deviations of swim properties, such as `IBI pitch`, `peak pitch`, `trajectory`, `end pitch`, `end angVel`, etc.
6000 bouts should give you robust regression for Righting, Steering, and Timing. Fin-body ratio may require close to 10,000 bouts to converge.
Check the SAMPL manuscript for bout number required to resolve differences in kinematic parameters.

However, if the bout number per repeat is too low, variations of swim properties among fish from different clutches may become dominant and introduce inconsistency. Regression-based parameters may not converge easily, and even the standard deviation of parameters can vary quite a lot. In this case, segmenting/categorizing swim bouts and calculating the means may help. See the section below for details.

## What should I look at if I don't have enough bouts for regression analysis?

If bout number per condition/repeat is not ideal for regression analysis, one might look into the mean of basic bout properties. However, it is important to note that mean of bout properties works the best when bouts are properly categorized.
We recommende categorizing swim bouts into 2 or 4 categories by righting rotation and steering rotation.

1. Categorization by righting rotation or initial pitch

Fish adopt different strategies for nose-up and nose-down bouts. We found that the direction of righting rotation serves as a good indicator for nose-up/down bouts. We would recommend first segmenting all bouts into positive & negative righting rotations. Alternatively, you may separate bouts by pitch initial (-250ms) at 10deg or so, which usually yields similar results.

2. Categorization by righting and steering

By plotting angular velocity vs time, we found that bouts can be categorized into 4 types by the directions they rotate toward: Steering(+) Righting(+), Steering(–) Righting(–), Steering(+) Righting(–), and Steering(–) Righting(+).

Therefore, to further categorize bouts, we suggest separating by positive and netagive steering rotations.

3. Parameters worth checking

Below is a list of parameters that we suggest to check the mean of, for each bout categories:
    `trajectory`: direction of movement at the time of peak speed
    `peak posture`: posture at the time of peak speed
    `initial posture`: posture at -250ms
    `end posture`: posture at 200ms
    `initial angvel`: angular velocity at -250ms
    `end angvel`: angular velocity at 200ms
    `pitch change` or `total rotation`: posture change from -250 to 200ms
    `angvel change`: angular velocity change from -250 to 200ms
    `trajectory pre phased`: average direction of movement from -250ms to -100ms
    `righting rotation` and `steering rotation`

## Single fish dilemma

A very useful parameter is the standard deviation of pitch angles, which represents how well the fish are able to maintain their posture.
However, due to individual variations and slight differences between apparatuses (which we will cover later), different fish from different boxes may yield similar STD but different MEANs.
When pooling bouts from multiple fish/boxes, STD is almost certain to be wider than that of bouts collected from a single fish/single box.

One way to overcome this is to concatenate bouts from multiple fish with similar levels of manipulation (# of cells activated/ablated, fluorescent levels etc.).
However, if you want to run individual fish and determine treatment effectiveness in a fish-by-fish base, we recommend 
