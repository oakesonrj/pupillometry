from mne.io import read_raw_eyelink
from mne import create_info, EpochsArray
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


            ##### LOADING IN DATA, SOME PARTICIPANTS ONLY HAVE PART A ####### using the .fif from the previous step

subj="07"

raw_fif = f'{subj}.fif'

raw_fif = mne.io.read_raw_fif(raw_fif)

print(raw_fif.annotations)

events, event_id = mne.events_from_annotations(raw_fif)

print("Event IDs:", event_id)  # e.g., {'condition_1': 1, 'condition_2': 2, ...}
print("Events:", events[:]) 


# Initialize a dictionary to store onset and offset times for each condition
condition_times = {'target only': [], 'target_only': [], '1_speaker': [], '2_speaker': [], 'SSN': []} #accidently had named different conditions target_only and "target only"

# Loop through annotations to group onsets and offsets by condition
for desc, onset, duration in zip(raw_fif.annotations.description, raw_fif.annotations.onset, raw_fif.annotations.duration):
    for condition in condition_times:
        if condition + '_onset' in desc:
            print(desc, onset, float(duration))
            condition_times[condition].append((onset, onset + duration))

# Verify extracted times
for condition, times in condition_times.items():
    print(f"{condition}: {times}")

# Define a mapping to unify equivalent condition names in case mismatch names in annotations
standardized_conditions = {
    'target_only': 'target_only',
    'target only': 'target_only',
    '1_speaker': '1_speaker',
    '2_speaker': '2_speaker',
    'SSN': 'SSN'
}

# Initialize a dictionary to store onset, offset times, and durations
condition_times = {key: {'onsets': [], 'offsets': [], 'durations': []} for key in standardized_conditions.values()}

# Process annotations with standardized names
for desc, onset, duration in zip(raw_fif.annotations.description, raw_fif.annotations.onset, raw_fif.annotations.duration):
    # Map the description to the standardized condition name
    for original, standardized in standardized_conditions.items():
        if original + '_onset' in desc:
            condition_times[standardized]['onsets'].append(onset)
        elif original + '_offset' in desc:
            condition_times[standardized]['offsets'].append(onset)

# Predefined durations for synthetic offsets in target_only or if a condition is missing an offset in the .fif file
predefined_durations = {3: 30.0, 4: 30.0, 6: 60.0}  # Example: adjust as needed, in this case primarly used for the 4th and 7th target_only conditions that are missing offsets

# Process each condition
for condition, times in condition_times.items():
    onsets = times['onsets']
    offsets = times['offsets']
    paired_durations = []

    if condition == 'target_only':  # Handle target_only with synthetic offsets
        synthesized_offsets = []
        i_on, i_off = 0, 0
        while i_on < len(onsets):
            onset = onsets[i_on]
            if i_off < len(offsets):
                offset = offsets[i_off]
                duration = offset - onset
                if duration > predefined_durations.get(i_on, 60) * 1.1:  # Check for mismatch, we take the onset and synthesize an offset
                    synthetic_offset = onset + predefined_durations.get(i_on, 60)
                    synthesized_offsets.append(synthetic_offset)
                    paired_durations.append(predefined_durations.get(i_on, 60))
                else:
                    synthesized_offsets.append(offset)
                    paired_durations.append(duration)
                    i_off += 1
            else:  # If no more offsets available, synthesize one
                synthetic_offset = onset + predefined_durations.get(i_on, 60)
                synthesized_offsets.append(synthetic_offset)
                paired_durations.append(predefined_durations.get(i_on, 60))
            i_on += 1

        times['offsets'] = synthesized_offsets
        times['durations'] = paired_durations

    else:  # Handle straightforward conditions
        for i, onset in enumerate(onsets):
            if i < len(offsets):
                offset = offsets[i]
                paired_durations.append(offset - onset)
            else:
                print(f"Unmatched onset for condition {condition} at index {i}: {onset}")
        times['durations'] = paired_durations

# Display final results
for condition, times in condition_times.items():
    print(f"{condition}:")
    print(f"  Onsets: {times['onsets']}")
    print(f"  Offsets: {times['offsets']}")
    print(f"  Durations: {times['durations']}")

# Pick the pupil channel(s) (adjust based on your channel names)
pupil_data = raw_fif.copy().pick_channels(['pupil_right']).get_data()[0]
time = raw_fif.times  # Time vector

# Compute basic statistics
mean_pupil = np.mean(pupil_data)
std_pupil = np.std(pupil_data)
min_pupil = np.min(pupil_data)
max_pupil = np.max(pupil_data)

# Create a Pandas DataFrame for more insights
pupil_df = pd.DataFrame({'Time (s)': time, 'Pupil Size': pupil_data})
stats = pupil_df.describe()  # Get detailed stats

print(f"Mean: {mean_pupil}, Std: {std_pupil}, Min: {min_pupil}, Max: {max_pupil}")
print(stats)

plt.figure(figsize=(15, 5))
plt.plot(time, pupil_data, label="Pupil Size")
plt.xlabel("Time (s)")
plt.ylabel("Pupil Size")
plt.title("Pupil Size Trend Over 21 Minutes")
plt.legend()
plt.show()

# shows annotations of interest by hiding other annotations
df = raw_fif.annotations.to_data_frame()
idx = np.array(list(df.description != 'saccade')) * np.array(list(df.description != 'fixation')) * np.array(list(df.description != 'BAD_'))* np.array(list(df.description != 'BAD_blink'))
df.iloc[idx]

#save new version of .fif
new_file = f"{subj}.fif"
raw_fif.save(new_file, overwrite=True)
