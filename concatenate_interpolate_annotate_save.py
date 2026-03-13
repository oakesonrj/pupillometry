from mne.io import read_raw_eyelink
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.interpolate import CubicSpline,interp1d,PchipInterpolator,Akima1DInterpolator
import math, mne
from nih2mne.eyetracking_preprocessing import *
# %matplotlib qt


                  ##### LOADING IN DATA, SOME PARTICIPANTS ONLY HAVE PART A ####### adjust as needed

subj="C101"

part_A = glob.glob(f'/data/MEGLANG/SPIN/Raw_data/SPIN_ASCs/{subj}PA*.asc')
part_B = glob.glob(f'/data/MEGLANG/SPIN/Raw_data/SPIN_ASCs/{subj}PB*.asc')
rest = glob.glob(f'/data/MEGLANG/SPIN/Raw_data/SPIN_ASCs/{subj}_Rest_*.asc')

raw_A = read_raw_eyelink(part_A[0], create_annotations=True)
raw_B = read_raw_eyelink(part_B[0], create_annotations=True)

if rest:    
    raw_rest = read_raw_eyelink(rest[0], create_annotations=True)
    print("Rest loaded from eye-tracker")
else:
    print("!!! No pupillometry rest dataset for participant, check ADC !!!")

'''
# Check if part B exists, then load it
if os.path.exists(part_B):
    raw_B = read_raw_eyelink(part_B, create_annotations=True)
    raw_rest = read_raw_eyelink(rest, create_annotations = True)
    
    raw_total = mne.concatenate_raws([raw_A, raw_B])
else:
    print(f'Part B for subject {subj} not found. Proceeding with Part A only.')
    print(f'Rest for subject {subj} not found.')
    raw_total = raw_A  # Use only part A if part B doesn't exist
'''
raw_total = mne.concatenate_raws([raw_A, raw_B])
info = raw_total.info
# print(f'sampling rate was {info["sfreq"]}Hz')
# print(f'data contains {raw_total._data.shape[0]} channels and {raw_total._data.shape[1]} timepoints')
# print(f'channel names are {[ch for ch in raw_total.ch_names]}')

# taking the annotated raw data
annot_df = raw_total.annotations.to_data_frame()
# annot_dfB = raw_B.annotations.to_data_frame()

task_onset_pupil = raw_total.annotations.onset[annot_df.description=='CONDITION target only'][0]
task_offset_pupil = raw_total.annotations.onset[annot_df.description=='CONDITION target_only'][2]+60

# task_onset_pupilB = raw_B.annotations.onset[annot_dfB.description=='CONDITION target_only'][0]
# task_offset_pupilB = raw_B.annotations.onset[annot_dfB.description=='CONDITION target_only'][2]+60

cropped_raw_pupil = raw_total.copy().crop(tmin=task_onset_pupil, tmax=task_offset_pupil)
# cropped_raw_pupilB = raw_B.copy().crop(tmin=task_onset_pupilB, tmax=task_offset_pupilB)

pupil_data = cropped_raw_pupil._data[cropped_raw_pupil.ch_names.index('pupil_right')]
# pupil_dataB = cropped_raw_pupilB._data[cropped_raw_pupilB.ch_names.index('pupil_right')]


#%% SPLINE INTERPOLATION part A###########################################################

                    ######## LABELING SACCADE START AND END POINTS #######
                    
# blink annotations
blink_annotations = annot_df[annot_df.description=='BAD_blink']
# saccade annotations
saccade_annotations = annot_df[annot_df.description=='saccade']


# Convert 'onset' and 'duration' to seconds if they are in datetime format
if pd.api.types.is_datetime64_any_dtype(saccade_annotations['onset']):
    saccade_annotations.loc[:, 'onset'] = (saccade_annotations['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(saccade_annotations['duration']):
    saccade_annotations.loc[:, 'duration'] = saccade_annotations['duration'] / pd.Timedelta(seconds=1)

saccade_starts = (saccade_annotations['onset'] * cropped_raw_pupil.info['sfreq']).astype(int)
saccade_ends = (saccade_starts + saccade_annotations['duration'] * cropped_raw_pupil.info['sfreq']).astype(int)
print(saccade_starts, saccade_ends)
saccade_duration = np.array(saccade_ends) - np.array(saccade_starts)


if pd.api.types.is_datetime64_any_dtype(blink_annotations['onset']):
    blink_annotations.loc[:, 'onset'] = (blink_annotations['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(blink_annotations['duration']):
    blink_annotations.loc[:, 'duration'] = blink_annotations['duration'] / pd.Timedelta(seconds=1)

blink_starts = (blink_annotations['onset'] * cropped_raw_pupil.info['sfreq']).astype(int)
blink_ends = (blink_starts + blink_annotations['duration'] * cropped_raw_pupil.info['sfreq']).astype(int)
print(blink_starts, blink_ends)
blink_duration = np.array(blink_ends) - np.array(blink_starts)


all_saccade_interp = cropped_raw_pupil.copy()
#all_saccade_interp = raw_total.crop(tmin=first_target_only,tmax=None)

#% Thresholded Saccade Interpolation

#print(saccade_duration)
print(f"saccade_duration type: {type(saccade_duration)}, shape: {np.shape(saccade_duration)}")
for long_duration in saccade_duration:
    if long_duration >= 83:
        print(f"long saccade duration in samples: {long_duration}")
annot_df = all_saccade_interp.annotations.to_data_frame()
elapsed_time = (annot_df['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

# Print the index and value for events greater than or equal to 0.083
print(len(saccade_annotations))

long_saccade_list = []
for i, long_saccade in saccade_annotations['duration'].items():
    if long_saccade >= 0.083:
        print(f"Event: {i}, Duration in seconds: {long_saccade}; and Elapsed time: {elapsed_time[i]}")
        long_saccade_list.append(long_saccade)
# Define the duration threshold for long saccades in samples
duration_threshold = 83

# Calculate the saccade durations in samples
saccade_duration = np.array(saccade_ends) - np.array(saccade_starts)


# Loop through all saccades and apply the window and interpolation only for long saccades
for idx in range(len(saccade_duration)):

    # Check if the current saccade's duration is greater than or equal to the threshold
    if saccade_duration[idx] >= duration_threshold:

        # Define window around the saccade (you can adjust this window as needed)
        window = (-100, 100)  #  100 samples before the start and 100 samples after the end, adjust as needed

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
        
        anchor = 50
        
        # Define any number of points around the start and end for interpolation, adjust as needed
        start_points_x = np.arange(max(0, start - anchor), start)
        #start_points_x = np.array([start - 12, start - 10, start - 8, start - 6, start - 4, start - 2])
        start_points_x = start_points_x[(start_points_x >= 0) & (start_points_x < len(pupil_data))]
        start_points_y = pupil_data[start_points_x]

        # Define points after the end
        end_points_x = np.arange(end, min(len(pupil_data), end + anchor))
        #end_points_x = np.array([end + 2, end + 4, end + 6, end + 8, end + 10, end + 12])
        end_points_x = end_points_x[(end_points_x >= 0) & (end_points_x < len(pupil_data))]
        end_points_y = pupil_data[end_points_x]

        # Combine start and end points for interpolation
        interp_x = np.concatenate([start_points_x, end_points_x])
        interp_y = np.concatenate([start_points_y, end_points_y])

        # Generate the cubic spline interpolator and apply over the section
        cs = CubicSpline(interp_x, interp_y)
        new_section1 = cs(interp_points)
        
        # Pchip interpolator
        pchip = PchipInterpolator(interp_x, interp_y)
        new_section2 = pchip(interp_points)
        
        # akima interpolator
        # akima = Akima1DInterpolator(interp_x, interp_y)
        # new_section3 = akima(interp_points)
        
        #makima interpolator
        makima = Akima1DInterpolator(interp_x, interp_y, method = "makima")
        new_section4 = makima(interp_points)
        
        # Replace the interpolated data in the original data array
        try:
            all_saccade_interp._data[:, start:start + len(new_section2)] = new_section2
        except Exception as e:
            print(f"Error processing saccade #{idx}: {e}")


print(start, end)
print(window)

#%%filter part A for annotations
cropped_total_filt = all_saccade_interp.copy().filter(None, 5)
cropped_total_filt.pick('pupil_right').plot(title=subj + " Pupil w/ Pchip Interpolation: Part A")

# Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_duration = cropped_raw_pupil.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in cropped_total_filt.annotations if annot['description'] == 'BAD_blink'])
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)

#%% part B: check new annotations and interactive plot with new annotations
df = cropped_total_filt.annotations.to_data_frame()
df[df['description']=='BAD_remove'] # or whatever BAD_ has been renamed to

cropped_annot = all_saccade_interp.copy().filter(None, 5)
cropped_annot.set_annotations(cropped_total_filt.annotations)

print(len(cropped_total_filt.annotations), len(cropped_raw_pupil.annotations))
cropped_annot.pick('pupil_right').plot()


#%% part B SAME AS LINE 196 - Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_duration = cropped_total_filt.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in cropped_total_filt.annotations if annot['description'] == 'bad_remove'])
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)



#%%
old_dir, old_name = os.path.split(str(cropped_annot.filenames[0]))
new_name = os.path.join(old_dir, f"{subj}_pupilannot_raw_partB.fif")
print(new_name)

cropped_annot.save(new_name, overwrite=True)
