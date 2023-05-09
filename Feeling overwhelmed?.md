# Navigate SAMPL data analysis

## When should I look at my data?

From our experience, any experiment with 4 or more *standard* repeats should yield enough data for pilot analysis. 
One standard repeat is defined as a 48-hour dataset collected from 3 boxes/condition, with each box having 2-3 fish/narrow chamber or 5-8 fish/standard chamber.
This should give you more than 6000 bouts per condition, which is good enough to compare standard deviations of swim properties, such as `IBI pitch`, `peak pitch`, `end pitch`, `end angVel`, etc.
6000 bouts should give you robust regression for Righting, Steering, and Timing. Fin-body ratio may require close to 10,000 bouts to converge.
If bout number is limited, see section below for tips on analysis.






## Single fish dilemma

A typical parameter we look at is standard deviation of pitch angles, which represents how well the fish are able to maintain their posture.
However, due to individual variations and slight differences between apparatuses (which we will cover later), different fish from different boxes may yield similar STD but different MEANs.
When pooling bouts from multiple fish/boxes, STD is almost certain to be wider than that of bouts collected from a single fish/single box.
However, for some experiments, one may want to run individual fish and determine treatment effectiveness in a fish-by-fish base.
