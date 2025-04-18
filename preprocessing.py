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

subj=""

part_A = #filepath
part_B = #filepath

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

# Remove blinks by setting them to zero

for idx, annot in enumerate(raw_total.annotations):
    if annot['description'] == 'BAD_blink':
        start_sample = int(annot['onset'] * raw_total.info['sfreq']) - 20
        end_sample = int(annot['onset'] * raw_total.info['sfreq']) + int(annot['duration'] * raw_total.info['sfreq']) + 20
        raw_total._data[:, start_sample:end_sample] = 0  # Zero out the blink period


# Confirm that blinks were removed
print("Blinks have been zeroed out.")


                                                    ################################################## CUBIC SPLINE INTERPOLATION ###########################################################

                    ######## LABELING BLINK START AND END POINTS #######

pupil_data = raw_total._data[raw_total.ch_names.index('pupil_right')]

# get blink annotations
annot_df = raw_total.annotations.to_data_frame()
blink_annotations = annot_df[annot_df.description=='BAD_blink']
#blink_annotations

blink_annotations.iloc[0]['onset']

# Convert 'onset' and 'duration' to seconds if they are in datetime format
if pd.api.types.is_datetime64_any_dtype(blink_annotations['onset']):
    blink_annotations.loc[:, 'onset'] = (blink_annotations['onset'] - annot_df.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(blink_annotations['duration']):
    blink_annotations.loc[:, 'duration'] = blink_annotations['duration'] / pd.Timedelta(seconds=1)

blink_starts = (blink_annotations['onset'] * raw_total.info['sfreq']).astype(int)
blink_ends = (blink_starts + blink_annotations['duration'] * raw_total.info['sfreq']).astype(int)


                    ####### PERFORM CUBIC SPLINE USING START AND END POINTS ########
    
    # Define the indices for interpolation

for idx in range(len(blink_starts)):
    window = (-50, 50)
    start, end = np.array(blink_starts)[idx] + window[0] , np.array(blink_ends)[idx] + window[1] 

    # Define the indices for interpolation
    interp_points = np.arange(start, end)
    orig_data_section = pupil_data[interp_points]

    # Define points around the start and end for interpolation - Two (OR MORE) points before the start (otherwise running linear interpolation, see below) 
    start_points_x = np.array([start - 12, start - 10, start - 9])
    start_points_y = pupil_data[start_points_x]

    # Two points (OR MORE) after the end
    end_points_x = np.array([end + 9, end + 10, end + 12])
    end_points_y = pupil_data[end_points_x]

    # Combine start and end points for interpolation
    interp_x = np.concatenate([start_points_x, end_points_x])
    interp_y = np.concatenate([start_points_y, end_points_y])

    # Generate the cubic spline interpolator
    cs = CubicSpline(interp_x, interp_y)

    # Apply the cubic spline to interpolate over the section
    new_section = cs(interp_points)
    
    # In case the first event is a bad_blink
    try:
        raw_total._data[:,start:start+len(new_section)]=new_section
    except:
        pass


# plot entire task with only the pupil channel with interpolation and a low-pass filter

raw_total.copy().filter(None, 20).pick('pupil_right').plot()


##############################################################################################################
############ LINEAR INTERPOLATION (IF NEEDED OR PREFERRED) #############
orig_data_section = pupil_data[np.arange(start,end)]
print(start, end)
x = np.array([start, end])
y = np.array([pupil_data[start], pupil_data[end]])
    
    # Generate the spline interpolator
cs = CubicSpline(x, y)

new_section = cs(np.arange(start, end))

fig, ax=plt.subplots()

ax.plot(raw_total.times[np.arange(start,end)],orig_data_section)
ax.plot(raw_total.times[np.arange(start,end)], new_section, ls='--')
###########################################################
#############################################################################################################


                                                      ###################################################### DICTIONARY TO LEAD TO EPOCHING ###########################################################
# print annotation-related info
if raw_total.annotations:
    annot_df = raw_total.annotations.to_data_frame()
    print(f'{len(annot_df)} annotations were found')
    unique_annots = set(annot_df.description.to_list())
    print(f'detected annotation names were: {unique_annots}')

# Predefined mapping for specific onset keys
specific_values = {
    'condition1_speaker_onset': 2,
    'condition2_speaker_onset': 3,
    'conditionSSN_onset': 4,
    'conditiontarget_only_onset': 1,
    'conditiontarget only_onset': 1
}

# Initialize the dictionary to store the filtered events with specific values
filtered_events = {}

# Loop through the original set of events
for annotations in unique_annots:
    if 'onset' in annotations:
        print(annotations)
        # Extract the event label before the timestamp
        event_label = annotations.split(' | ')[0]
        
        # Check if the event label is in the specific_values dictionary
        if event_label in specific_values:
            filtered_events[annotations] = specific_values[event_label]
            

# Output the filtered dictionary with specific onset events and their assigned specific values
#print(filtered_events)

len(raw_total.annotations.__dict__['description'])

for key,val in filtered_events.items():
    print(key,val)

# make clean version of event dictionary
event_dict_clean = {'target_only':1, 
                    'onespeaker':2,
                    'twospeaker':3,
                    'ssn':4}

# CHUNK DATA
epochs = mne.Epochs(raw_total, events=events, 
                    event_id=event_dict_clean, 
                    tmin=0.001, tmax=30,
                    on_missing='warn',
                    reject_by_annotation=False,
                    reject=None,
                    baseline = None, preload=True) # no baseline correction


for count in range(len(events)):
    print(events[count,-1], epochs[count].event_id)


# check that I got the correct number of events  per condition

for key in event_dict_clean.keys():
    print(key, f"number of events: {epochs[key].events.shape[0]}")

# event Extraction
events, _ = mne.events_from_annotations(raw_total, event_id=filtered_events)

# look at data: # trials x channels x time
epochs['target_only'].get_data().shape 
data = epochs['target_only'].get_data()
fig, ax=plt.subplots(3,1)
for chI in range(3):
    ax[chI].plot(data[:,chI,:].T)
plt.show()

epochs['target_only'].drop_log

#################################################################################################################################################







