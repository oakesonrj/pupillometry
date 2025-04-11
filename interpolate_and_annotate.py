from mne.io import read_raw_eyelink
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.interpolate import CubicSpline,interp1d
import math, mne
from eyetracking_preprocessing import *
%matplotlib qt


                  ##### LOADING IN DATA, SOME PARTICIPANTS ONLY HAVE PART A ####### adjust as needed

subj="07"

part_A = f'{subj}.asc'
part_B = f'{subj}.asc'

raw_A = read_raw_eyelink(part_A, create_annotations = True)

# Check if part B exists, then load it
if os.path.exists(part_B):
    raw_B = read_raw_eyelink(part_B, create_annotations=True)
    # Concatenate A and B
    raw_total = mne.concatenate_raws([raw_A, raw_B])
else:
    print(f'Part B for subject {subj} not found. Proceeding with Part A only.')
    raw_total = raw_A  # Use only part A if part B doesn't exist

info = raw_total.info
print(f'sampling rate was {info["sfreq"]}Hz')
print(f'data contains {raw_total._data.shape[0]} channels and {raw_total._data.shape[1]} timepoints')
print(f'channel names are {[ch for ch in raw_total.ch_names]}')




              ########################################### REMOVE BLINKS ########################################

# Remove blinks by setting them to zero
'''
for idx, annot in enumerate(raw_total.annotations):
    if annot['description'] == 'BAD_blink':
        start_sample = int(annot['onset'] * raw_total.info['sfreq']) - 20
        end_sample = int(annot['onset'] * raw_total.info['sfreq']) + int(annot['duration'] * raw_total.info['sfreq']) + 20
        raw_total._data[:, start_sample:end_sample] = 0  # Zero out the blink period

annot_df = raw_total.annotations.to_data_frame()
elapsed_time = (annot_df['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

# Print the index and value for events greater than or equal to 0.100
print(len(saccade_annotations))

long_saccade_list= []
for i, long_saccade in saccade_annotations['duration'].items():
    if long_saccade >= 0.083:
        print(f"Event: {i}, Duration: {long_saccade}; Elapsed time: {elapsed_time[i]}")
        long_saccade_list.append(long_saccade)
'''



# Confirm that blinks were removed
#print("Blinks have been zeroed out.")


################################################## CUBIC SPLINE INTERPOLATION ###########################################################

                    ######## LABELING SACCADE START AND END POINTS #######

pupil_data = raw_total._data[raw_total.ch_names.index('pupil_right')]

# taking the annotated raw data
annot_df = raw_total.annotations.to_data_frame()
# blink annotations
blink_annotations = annot_df[annot_df.description=='BAD_blink']
# saccade annotations
saccade_annotations = annot_df[annot_df.description=='saccade']


# Convert 'onset' and 'duration' to seconds if they are in datetime format
if pd.api.types.is_datetime64_any_dtype(saccade_annotations['onset']):
    saccade_annotations.loc[:, 'onset'] = (saccade_annotations['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(saccade_annotations['duration']):
    saccade_annotations.loc[:, 'duration'] = saccade_annotations['duration'] / pd.Timedelta(seconds=1)

saccade_starts = (saccade_annotations['onset'] * raw_total.info['sfreq']).astype(int)
saccade_ends = (saccade_starts + saccade_annotations['duration'] * raw_total.info['sfreq']).astype(int)
print(saccade_starts, saccade_ends)
saccade_duration = np.array(saccade_ends) - np.array(saccade_starts)


if pd.api.types.is_datetime64_any_dtype(blink_annotations['onset']):
    blink_annotations.loc[:, 'onset'] = (blink_annotations['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(blink_annotations['duration']):
    blink_annotations.loc[:, 'duration'] = blink_annotations['duration'] / pd.Timedelta(seconds=1)

blink_starts = (blink_annotations['onset'] * raw_total.info['sfreq']).astype(int)
blink_ends = (blink_starts + blink_annotations['duration'] * raw_total.info['sfreq']).astype(int)
print(blink_starts, blink_ends)
blink_duration = np.array(blink_ends) - np.array(blink_starts)

blink_saccade_interp = raw_total.copy()

all_saccade_interp = raw_total.copy()

mne_interp = raw_total.copy()

### Thresholded Saccade Interpolation

#print(saccade_duration)
print(f"saccade_duration type: {type(saccade_duration)}, shape: {np.shape(saccade_duration)}")
for long_duration in saccade_duration:
    if long_duration >= 41.5:
        print(f"long saccade duration in samples: {long_duration}")
annot_df = all_saccade_interp.annotations.to_data_frame()
elapsed_time = (annot_df['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

# Print the index and value for events greater than or equal to 0.083
print(len(saccade_annotations))

long_saccade_list= []
for i, long_saccade in saccade_annotations['duration'].items():
    if long_saccade >= 0.083:
        print(f"Event: {i}, Duration in seconds: {long_saccade}; and Elapsed time: {elapsed_time[i]}")
        long_saccade_list.append(long_saccade)
# Define the duration threshold for long saccades in samples
duration_threshold = 41.5

# Calculate the saccade durations in samples
saccade_duration = np.array(saccade_ends) - np.array(saccade_starts)


# Loop through all saccades and apply the window and interpolation only for long saccades
for idx in range(len(saccade_duration)):

    # Check if the current saccade's duration is greater than or equal to the threshold
    if saccade_duration[idx] >= duration_threshold:

        # Define window around the saccade (you can adjust this window as needed)
        window = (-30, 30)  # 30 samples before the start and 30 samples after the end, adjust as needed

        start = np.array(saccade_starts)[idx] + window[0]
        end = np.array(saccade_ends)[idx] + window[1]

        # Ensure the window stays within bounds of the pupil data
        if start < 0:
            print(f"Saccade #{idx}: Start of window is before data range; setting start to 0.")
            start = 0

        if end > len(pupil_data):
            print(f"Saccade #{idx}: End of window is larger than data; reducing window to last datapoint.")
            end = len(pupil_data) - 13

        # Define the interpolation range
        interp_points = np.arange(start, end)

        # Define any number of points around the start and end for interpolation, adjust as needed
        start_points_x = np.array([start - 12, start - 9, start - 6, start - 3])
        start_points_x = start_points_x[(start_points_x >= 0) & (start_points_x < len(pupil_data))]
        start_points_y = pupil_data[start_points_x]

        # Define points after the end
        end_points_x = np.array([end + 3, end + 6, end + 9, end + 12])
        end_points_x = end_points_x[(end_points_x >= 0) & (end_points_x < len(pupil_data))]
        end_points_y = pupil_data[end_points_x]

        # Combine start and end points for interpolation
        interp_x = np.concatenate([start_points_x, end_points_x])
        interp_y = np.concatenate([start_points_y, end_points_y])

        # Generate the cubic spline interpolator
        cs = CubicSpline(interp_x, interp_y)

        # Apply the cubic spline to interpolate over the section
        new_section = cs(interp_points)

        # Replace the interpolated data in the original data array
        try:
            all_saccade_interp._data[:, start:start + len(new_section)] = new_section
        except Exception as e:
            print(f"Error processing saccade #{idx}: {e}")


print(start, end)
print(window)

raw_total_filt = all_saccade_interp.copy().filter(None, 20)
raw_total_filt.pick('pupil_right').plot()

# Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_duration = raw_total.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in raw_total.annotations if annot['description'] == 'BAD_blink'])
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)

# check occurances of a specific annotation
df = raw_total_filt.annotations.to_data_frame()
df[df['description']=='BAD_']

# saving as a .fif, with new annotations
old_dir, old_name = os.path.split(str(raw_total_annot.filenames[0]))
new_name = os.path.join(old_dir, f"{subj}_pupilannot.fif")
print(new_name)

raw_total_annot.save(new_name, overwrite=True)

# SAME AS LINE 196 - Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_duration = raw_total.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in raw_total_annot.annotations if annot['description'] == 'BAD_']) # different description here than above
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)




