*can ignore "preprocessing.py"

The following code is task specific and related mainly to pupillometry. Be mindful of where to change things e.g., loading in data, onsets, offsets, durations, and adding synthetic offsets if needed.

This code will:
- Load in .asc files (if you are using eyelink and SR Research, they have and EDF to ASC converter available)
- Interpolate blinks and artifactual saccades
- Create .fif files which are used for annotation (visual inspection will be necessary)
- Epoch data
- Convert bad portions of data into NaN
- Provide comparisons of raw and cleaned pupil data


!! Several portions have overlapping code and potential variable names !! 


1) Saccade interpolation code experiments with various types of interpolations, either MNE's linear interpolation or two different cubic spline interpolations
     -- if you're comfortable/confident with the cubic spline that we implemented, feel free to jump to step 2

2) After running the code to line 193, this is where visual inspection and manual annotation will take place. Will save as a .fif
     -- in the plot screen, press "a" to open the annotations pop-up, create a new annotation "BAD_" (can be named whatever) and tick the scrollable option
     -- !! BUG WARNING !! it seems if you right click to erase an annotation made by mistake, there is a chance for MNE to delete all annotations, causing you to have to reload the data and start from the beginning. Be mindful of this

3) Load in the .fif file created in Step 2. Also does a lot of the things in Step 4.
     -- Create a dictionary for epoching
     -- will allow for inspection of onsets, offsets, and durations of events
     -- saves new .fif

4) Will convert BAD_ annotations to NaN then saves as a .csv
