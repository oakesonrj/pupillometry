from mne.io import read_raw_eyelink
from mne import create_info, EpochsArray
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.interpolate import CubicSpline,interp1d
from scipy.stats import linregress
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import math, mne
from eyetracking_preprocessing import *
%matplotlib qt


                                                                                  ##### LOADING IN DATA, SOME PARTICIPANTS ONLY HAVE PART A #######

subj="07"

raw_fif = f'C:\\Users\\oakesonrj\\OneDrive - National Institutes of Health\\Desktop\\PhD misc\\ASCs\\{subj}_pupilannot.fif'

raw_fif = mne.io.read_raw_fif(raw_fif)

print(raw_fif.annotations)

events, event_id = mne.events_from_annotations(raw_fif)

print("Event IDs:", event_id)  # e.g., {'condition_1': 1, 'condition_2': 2, ...}
print("Events:", events[:]) 

# Initialize a dictionary to store onset and offset times for each condition
condition_times = {'target only': [], 'target_only': [], '1_speaker': [], '2_speaker': [], 'SSN': []} #two target only's because i misnamed one accidently

# Loop through annotations to group onsets and offsets by condition
for desc, onset, duration in zip(raw_fif.annotations.description, raw_fif.annotations.onset, raw_fif.annotations.duration):
    for condition in condition_times:
        if condition + '_onset' in desc:
            print(desc, onset, float(duration))
            condition_times[condition].append((onset, onset + duration))

# Verify extracted times
for condition, times in condition_times.items():
    print(f"{condition}: {times}")

# Define a mapping to unify equivalent condition names in case mismatch names in annotations, refer back to the misname issue
standardized_conditions = {
    'target_only': 'target_only',
    'target only': 'target_only', #had to be renamed
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
for condition, times in condition_times.items():
    print(f"Condition: {condition}")
    print(f"Onsets: {times['onsets']}")
    print(f"Offsets: {times['offsets']}")

# Create a single figure for all segments
plt.figure(figsize=(15, 8))

for condition, times in condition_times.items():
    # Convert onsets and offsets to float if needed
    times["onsets"] = [float(o) for o in times["onsets"]]
    times["offsets"] = [float(o) for o in times["offsets"]]

    for onset, offset in zip(times["onsets"], times["offsets"]):
        condition_mask = (time >= onset) & (time <= offset)
        
        if np.any(condition_mask):
            # Perform regression for this segment
            slope, intercept, r_value, p_value, std_err = linregress(
                time[condition_mask], pupil_data[condition_mask]
            )
            regression_line = slope * time[condition_mask] + intercept

            # Plot the segment and its regression line
            plt.plot(
                time[condition_mask], 
                pupil_data[condition_mask], 
                label=f"{condition} {onset:.1f}s-{offset:.1f}s Pupil Size"
            )
            plt.plot(
                time[condition_mask], 
                regression_line, 
                linestyle="--", 
                label=f"{condition} {onset:.1f}s-{offset:.1f}s Regression"
            )

# Finalize and show the figure
plt.xlabel("Time (s)")
plt.ylabel("Pupil Size")
plt.title("Pupil Size Trends Across Conditions and Segments")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Move legend outside plot if too cluttered
plt.tight_layout()
plt.show()


#### now to convert the bad portions into NaN for stats and save as a .csv
# Step 1: Mask BAD_ annotations as NaN
bad_intervals = []

# Extract BAD_ annotations
for desc, onset, duration in zip(raw_fif.annotations.description, raw_fif.annotations.onset, raw_fif.annotations.duration):
    if 'BAD_' in desc:
        bad_intervals.append((onset, onset + duration))

# Create a mask for the pupil data to set BAD_ intervals to NaN
bad_mask = np.zeros_like(pupil_data, dtype=bool)
for start, end in bad_intervals:
    bad_mask |= (time >= start) & (time <= end)

# Set BAD_ sections in the pupil data to NaN
pupil_data_cleaned = pupil_data.copy()
pupil_data_cleaned[bad_mask] = np.nan

# Step 2: Extract data for the 4 conditions and prepare for CSV export
data_rows = []  # Collect rows for the CSV file
for condition, times in condition_times.items():
    for onset, offset in zip(times['onsets'], times['offsets']):
        condition_mask = (time >= onset) & (time <= offset)
        # Exclude NaN sections within each condition
        valid_indices = condition_mask & ~bad_mask
        valid_times = time[valid_indices]
        valid_pupil_sizes = pupil_data_cleaned[valid_indices]
        
        # Append data for CSV
        for t, size in zip(valid_times, valid_pupil_sizes):
            data_rows.append({'Time (s)': t, 'Pupil Size': size, 'Condition': condition})

# Step 3: Convert to DataFrame and save to CSV
pupil_df_cleaned = pd.DataFrame(data_rows)
csv_path = f'{subj}.csv'
pupil_df_cleaned.to_csv(csv_path, index=False)

print(f"Cleaned data saved to {csv_path}")


# Visualization of cleaned data (NaNs removed)
plt.figure(figsize=(15, 5))

# Plot original data in light gray for reference
plt.plot(time, pupil_data, color='lightgray', label="Original Pupil Size (with BAD_)")

# Plot cleaned data (NaNs removed) in blue
plt.plot(time[~np.isnan(pupil_data_cleaned)], 
         pupil_data_cleaned[~np.isnan(pupil_data_cleaned)], 
         color='blue', label="Cleaned Pupil Size (BAD_ removed)")

# Highlight BAD_ intervals (optional)
for start, end in bad_intervals:
    plt.axvspan(start, end, color='red', alpha=0.3, label="BAD_ Interval" if 'BAD_ Interval' not in plt.gca().get_legend_handles_labels()[1] else "")
# Step 1: Mask BAD_ annotations as NaN
bad_intervals = []

# Extract BAD_ annotations
for desc, onset, duration in zip(raw_fif.annotations.description, raw_fif.annotations.onset, raw_fif.annotations.duration):
    if 'BAD_' in desc:
        bad_intervals.append((onset, onset + duration))

# Create a mask for the pupil data to set BAD_ intervals to NaN
bad_mask = np.zeros_like(pupil_data, dtype=bool)
for start, end in bad_intervals:
    bad_mask |= (time >= start) & (time <= end)

# Set BAD_ sections in the pupil data to NaN
pupil_data_cleaned = pupil_data.copy()
pupil_data_cleaned[bad_mask] = np.nan

# Step 2: Extract data for the 4 conditions and prepare for CSV export
data_rows = []  # Collect rows for the CSV file
for condition, times in condition_times.items():
    for onset, offset in zip(times['onsets'], times['offsets']):
        condition_mask = (time >= onset) & (time <= offset)
        # Exclude NaN sections within each condition
        valid_indices = condition_mask & ~bad_mask
        valid_times = time[valid_indices]
        valid_pupil_sizes = pupil_data_cleaned[valid_indices]
        
        # Append data for CSV
        for t, size in zip(valid_times, valid_pupil_sizes):
            data_rows.append({'Time (s)': t, 'Pupil Size': size, 'Condition': condition})

# Step 3: Convert to DataFrame and save to CSV
pupil_df_cleaned = pd.DataFrame(data_rows)
csv_path = f'{subj}_pupil_data_cleaned.csv'
pupil_df_cleaned.to_csv(csv_path, index=False)

print(f"Cleaned data saved to {csv_path}")
# Add labels, legend, and title
plt.xlabel("Time (s)")
plt.ylabel("Pupil Size")
plt.title("Pupil Data Before and After Removing BAD_ Intervals")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
