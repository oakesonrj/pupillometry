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


                                                                                  ##### LOADING IN DATA, SOME PARTICIPANTS ONLY HAVE PART A #######

subj="49"

part_A = f'C:\\{subj}PA.asc' #file path
part_B = f'C:\\{subj}PB.asc' #in this case, had two parts needing concatenating

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




################################################## REMOVE BLINKS ########################################################################
###################### Probably not necessary, but here just in case #############

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

# Confirm that blinks were removed
#print("Blinks have been zeroed out.")
'''

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

### Raw copies to test different interpolations

all_saccade_interp = raw_total.copy() #interpolates around saccades set to a certain threshold thus also getting all blinks, helps with artifacts from long saccades, cubic spline

blink_saccade_interp = raw_total.copy() #only interpolates blinks, cubic spline

mne_interp = raw_total.copy() #mne base interpolation, linear 



##### Thresholded/All Saccade Interpolation #####

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
        window = (-30, 30)  # 50 samples before the start and 50 samples after the end

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

        # Define points around the start and end for interpolation
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



##### BLINK SACCADE INTERPOLATION #####
'''
for idx_blink in range(len(blink_annotations)):

    arr_blink = np.arange(blink_annotations.iloc[idx_blink].onset,blink_annotations.iloc[idx_blink].onset + blink_annotations.iloc[idx_blink].duration, 1/500)
    # get saccade associated to blinky

    for ii in range(len(saccade_annotations)):
        arr_saccade = np.arange(saccade_annotations.iloc[ii].onset,saccade_annotations.iloc[ii].onset + saccade_annotations.iloc[ii].duration, 1/500)
        #print(arr_blink, arr_saccade)
        bool_idx = np.isin(np.round(arr_blink,3), np.round(arr_saccade,3)) # Are all samples in blink appear in the saccade samples
        if bool_idx.sum() == len(arr_blink): 
            #print(f'for BLINK {idx_blink},  saccade {ii} is the one I want to use!!!')
            break

    # Check if the current saccade's duration is greater than or equal to the threshold
    if ii:

        # Define window around the saccade (you can adjust this window as needed)
        window = (-30, 30)  # 50 samples before the start and 50 samples after the end

        start = np.array(saccade_starts)[ii] + window[0]
        end = np.array(saccade_ends)[ii] + window[1]

        # Ensure the window stays within bounds of the pupil data
        if start < 0:
            print(f"Saccade #{ii}: Start of window is before data range; setting start to 0.")
            start = 0

        if end > len(pupil_data):
            print(f"Saccade #{ii}: End of window is larger than data; reducing window to last datapoint.")
            end = len(pupil_data) - 13

        # Define the interpolation range
        interp_points = np.arange(start, end)

        # Define points around the start and end for interpolation
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
            blink_saccade_interp._data[:, start:start + len(new_section)] = new_section
        except Exception as e:
            print(f"Error processing saccade #{ii}: {e}")


####
#### MNE interpolation (linear) #####

interpolate_pupil_mne = mne.preprocessing.eyetracking.interpolate_blinks(
    mne_interp, buffer=(0.05, 0.2), interpolate_gaze=True
)
print(interpolate_pupil_mne.ch_names)
####


### Compare interpolations ####
# Extract the 'pupil_right' data (assuming it is at index 2)
min_samples = min(
    blink_saccade_interp.get_data().shape[1],
    all_saccade_interp.get_data().shape[1],
    mne_interp.get_data().shape[1]
)
pupil_blink_saccade_interp = blink_saccade_interp.get_data()[2, :min_samples]
pupil_all_saccade_interp = all_saccade_interp.get_data()[2, :min_samples]
pupil_mne_interp = mne_interp.get_data()[2, :min_samples]

# Stack the data for all three interpolation methods along a new dimension (channels)
interp_data = np.stack([pupil_blink_saccade_interp, pupil_all_saccade_interp, pupil_mne_interp, raw_total._data[raw_total.ch_names.index('xpos_right')], raw_total._data[raw_total.ch_names.index('ypos_right')], raw_total._data[raw_total.ch_names.index('pupil_right')]])

# Define a sampling frequency
sfreq = 500  # Replace with actual sampling frequency of your data

# Create MNE Info structure with three channels, one per interpolation method
ch_names = ['pupil_blink_saccade_interp', 'pupil_all_saccade_interp', 'pupil_mne_interp', 'xpos_right', 'ypos_right', 'pupil_right']
ch_types = ['misc', 'misc', 'misc', 'misc', 'misc', 'misc']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
info.set_meas_date(raw_total.info['meas_date'])


# Create a RawArray with all three interpolation methods as separate channels
raw_interp = mne.io.RawArray(interp_data, info)
# grab annotations from raw data
annots = raw_total.annotations

# edit on which channels these annotations are applied to (in this case, to all channels)
for i in range(len(annots)):
    annots.ch_names[i] = tuple(ch_names)

# set the annotations to my new raw object
raw_interp.set_annotations(annots)
# Plot the data, using different colors for each channel
#raw_interp.plot()

raw_interp.pick(['pupil_right', 'pupil_blink_saccade_interp', 'pupil_all_saccade_interp', 'pupil_mne_interp']).plot()






###Interactive plot, press 'a' for annotations, create new draggable called BAD_ or whatever you want to name it e.g. BAD_pupil etc, deselect fixation, saccade and BAD_blink
raw_total_filt = all_saccade_interp.copy().filter(None, 20)
raw_total_filt.pick('pupil_right').plot()

# Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_duration = raw_total.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in raw_total.annotations if annot['description'] == 'BAD_blink'])
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)

### Original and interpolated data together to help with eyeballing good and bad interpolations
time = raw_total.times

plt.figure(figsize=(10, 6))

# Plot original pupil data (uninterpolated)
plt.plot(time, raw_total._data[raw_total.ch_names.index('pupil_right')], label='Uninterpolated', color='blue', alpha=0.6)

# Plot interpolated pupil data
plt.plot(time, all_saccade_interp._data[raw_total.ch_names.index('pupil_right')], label='All Saccade Interpolated', color='red', linestyle='--', alpha=0.6)
#plt.plot(time, all_saccade_interp._data[raw_total.ch_names.index('pupil_right')]- raw_total._data[raw_total.ch_names.index('pupil_right')], 
         #label='Diff', color='k', linestyle='solid', alpha=0.6)

#plt.plot(time, mne_interp._data[raw_total.ch_names.index('pupil_right')], label="MNE interpolation", color="cyan", alpha=0.6)

#plt.plot(time, blink_saccade_interp._data[raw_total.ch_names.index('pupil_right')], label="Blink interpolation", color="green", alpha=0.6)

plt.title(f'Subject {subj} - Pupil Data: Uninterpolated vs. Interpolated')
plt.xlabel('Time (seconds)')
plt.ylabel('Pupil Size')
plt.legend()
plt.show()


### check new annotations and interactive plot with new annotations
df = raw_total_filt.annotations.to_data_frame()
df[df['description']=='BAD_'] # or whatever BAD_ has been renamed to

raw_total_annot = all_saccade_interp.copy()
raw_total_annot.set_annotations(raw_total_filt.annotations)

print(len(raw_total_filt.annotations), len(raw_total.annotations))
raw_total_annot.pick('pupil_right').plot()

### save copy to .fif file
old_dir, old_name = os.path.split(str(raw_total_annot.filenames[0]))
new_name = os.path.join(old_dir, f"{subj}_pupilannot.fif")
print(new_name)

raw_total_annot.save(new_name, overwrite=True)

# like before, gives the percent of a give condition, i.e. BAD_blink, saccade, fixation, in this case the new annotations BAD_
total_duration = raw_total.times[-1]  # Total time in seconds
bad_durations = sum([annot['duration'] for annot in raw_total_annot.annotations if annot['description'] == 'BAD_']) # make sure the BAD_ label is consistent
percent_bad = (bad_durations / total_duration) * 100
print(percent_bad)
