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
#raw_total = mne.concatenate_raws([raw_A, raw_B])
#info = raw_total.info
# print(f'sampling rate was {info["sfreq"]}Hz')
# print(f'data contains {raw_total._data.shape[0]} channels and {raw_total._data.shape[1]} timepoints')
# print(f'channel names are {[ch for ch in raw_total.ch_names]}')

# taking the annotated raw data
annot_dfA = raw_A.annotations.to_data_frame()
annot_dfB = raw_B.annotations.to_data_frame()

task_onset_pupilA = raw_A.annotations.onset[annot_dfA.description=='CONDITION target only'][0]
#task_offset_pupilA = raw_A.annotations.onset[annot_dfA.description=='CONDITION target only'][3]+60

task_onset_pupilB = raw_B.annotations.onset[annot_dfB.description=='CONDITION target_only'][0]
#task_offset_pupilB = raw_B.annotations.onset[annot_dfB.description=='CONDITION target_only'][2]+60

cropped_raw_pupilA = raw_A.copy().crop(tmin=task_onset_pupilA, tmax=task_onset_pupilA) #+meg_len) can be modified to match with MEG data length
cropped_raw_pupilB = raw_B.copy().crop(tmin=task_onset_pupilB, tmax=task_onset_pupilB) #+meg_len)

pupil_dataA = cropped_raw_pupilA._data[cropped_raw_pupilA.ch_names.index('pupil_right')]
pupil_dataB = cropped_raw_pupilB._data[cropped_raw_pupilB.ch_names.index('pupil_right')]


#%% SPLINE INTERPOLATION part A###########################################################

                    ######## LABELING SACCADE START AND END POINTS #######
                    
# blink annotations
blink_annotationsA = annot_dfA[annot_dfA.description=='BAD_blink']
# saccade annotations
saccade_annotationsA = annot_dfA[annot_dfA.description=='saccade']


# Convert 'onset' and 'duration' to seconds if they are in datetime format
if pd.api.types.is_datetime64_any_dtype(saccade_annotationsA['onset']):
    saccade_annotationsA.loc[:, 'onset'] = (saccade_annotationsA['onset'] - annot_dfA.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(saccade_annotationsA['duration']):
    saccade_annotationsA.loc[:, 'duration'] = saccade_annotationsA['duration'] / pd.Timedelta(seconds=1)

saccade_startsA = (saccade_annotationsA['onset'] * cropped_raw_pupilA.info['sfreq']).astype(int)
saccade_endsA = (saccade_startsA + saccade_annotationsA['duration'] * cropped_raw_pupilA.info['sfreq']).astype(int)
print(saccade_startsA, saccade_endsA)
saccade_durationA = np.array(saccade_endsA) - np.array(saccade_startsA)


if pd.api.types.is_datetime64_any_dtype(blink_annotationsA['onset']):
    blink_annotationsA.loc[:, 'onset'] = (blink_annotationsA['onset'] - annot_dfA.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(blink_annotationsA['duration']):
    blink_annotationsA.loc[:, 'duration'] = blink_annotationsA['duration'] / pd.Timedelta(seconds=1)

blink_startsA = (blink_annotationsA['onset'] * cropped_raw_pupilA.info['sfreq']).astype(int)
blink_endsA = (blink_startsA + blink_annotationsA['duration'] * cropped_raw_pupilA.info['sfreq']).astype(int)
print(blink_startsA, blink_endsA)
blink_duration = np.array(blink_endsA) - np.array(blink_startsA)


all_saccade_interpA = cropped_raw_pupilA.copy()
#all_saccade_interp = raw_total.crop(tmin=first_target_only,tmax=None)

#% Thresholded Saccade Interpolation

#print(saccade_duration)
print(f"saccade_duration type: {type(saccade_durationA)}, shape: {np.shape(saccade_durationA)}")
for long_duration in saccade_durationA:
    if long_duration >= 83:
        print(f"long saccade duration in samples: {long_duration}")
annot_dfA = all_saccade_interpA.annotations.to_data_frame()
elapsed_timeA = (annot_dfA['onset'] - annot_dfA.iloc[0]['onset'])/pd.Timedelta(seconds=1)

# Print the index and value for events greater than or equal to 0.083
print(len(saccade_annotationsA))

long_saccade_listA = []
for i, long_saccade in saccade_annotationsA['duration'].items():
    if long_saccade >= 0.083:
        print(f"Event: {i}, Duration in seconds: {long_saccade}; and Elapsed time: {elapsed_timeA[i]}")
        long_saccade_listA.append(long_saccade)
# Define the duration threshold for long saccades in samples
duration_threshold = 83

# Calculate the saccade durations in samples
saccade_durationA = np.array(saccade_endsA) - np.array(saccade_startsA)


# Loop through all saccades and apply the window and interpolation only for long saccades
for idx in range(len(saccade_durationA)):

    # Check if the current saccade's duration is greater than or equal to the threshold
    if saccade_durationA[idx] >= duration_threshold:

        # Define window around the saccade (you can adjust this window as needed)
        window = (-100, 100)  #  100 samples before the start and 100 samples after the end, adjust as needed

        start = np.array(saccade_startsA)[idx] + window[0]
        end = np.array(saccade_endsA)[idx] + window[1]

        # Ensure the window stays within bounds of the pupil data
        if start < 0:
            print(f"Saccade #{idx}: Start of window is before data range; setting start to 0.")
            start = 0

        if end > len(pupil_dataA):
            print(f"Saccade #{idx}: End of window is larger than data; reducing window to last datapoint.")
            end = len(pupil_dataA) - 13

        # Define the interpolation range
        interp_points = np.arange(start, end)
        
        anchor = 50
        
        # Define any number of points around the start and end for interpolation, adjust as needed
        start_points_x = np.arange(max(0, start - anchor), start)
        #start_points_x = np.array([start - 12, start - 10, start - 8, start - 6, start - 4, start - 2])
        start_points_x = start_points_x[(start_points_x >= 0) & (start_points_x < len(pupil_dataA))]
        start_points_y = pupil_dataA[start_points_x]

        # Define points after the end
        end_points_x = np.arange(end, min(len(pupil_dataA), end + anchor))
        #end_points_x = np.array([end + 2, end + 4, end + 6, end + 8, end + 10, end + 12])
        end_points_x = end_points_x[(end_points_x >= 0) & (end_points_x < len(pupil_dataA))]
        end_points_y = pupil_dataA[end_points_x]

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
            all_saccade_interpA._data[:, start:start + len(new_section2)] = new_section2
        except Exception as e:
            print(f"Error processing saccade #{idx}: {e}")


print(start, end)
print(window)

#%%



#%%filter part A for annotations
cropped_filtA = all_saccade_interpA.copy().filter(None, 5)
cropped_filtA.pick('pupil_right').plot(title=subj + " Pupil w/ Pchip Interpolation: Part A")

# Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_durationA = cropped_filtA.times[-1]  # Total time in seconds
bad_durationsA = sum([annot['duration'] for annot in cropped_filtA.annotations if annot['description'] == 'BAD_blink'])
percent_badA = (bad_durationsA / total_durationA) * 100
print(percent_badA)

#%% part A: check new annotations and interactive plot with new annotations
dfA = cropped_filtA.annotations.to_data_frame()
dfA[dfA['description']=='BAD_remove'] # or whatever BAD_ has been renamed to

croppedA_annot = all_saccade_interpA.copy().filter(None, 5)
croppedA_annot.set_annotations(cropped_filtA.annotations)

print(len(cropped_filtA.annotations), len(cropped_raw_pupilA.annotations))
croppedA_annot.pick('pupil_right').plot()


#%% part A SAME AS LINE 341 - Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_durationA = cropped_filtA.times[-1]  # Total time in seconds
bad_durationsA = sum([annot['duration'] for annot in croppedA_annot.annotations if annot['description'] == 'bad_remove']) # different description here than above
percent_badA = (bad_durationsA / total_durationA) * 100
print(percent_badA)


#%% saving as a .fif, with new annotations
old_dir, old_name = os.path.split(str(croppedA_annot.filenames[0]))
new_name = os.path.join(old_dir, f"{subj}_pupilannot_raw_partA.fif")
print(new_name)

croppedA_annot.save(new_name, overwrite=True)

####

# PART B NEXT
#%%part B

# blink annotations
blink_annotationsB = annot_dfB[annot_dfB.description=='BAD_blink']
# saccade annotations
saccade_annotationsB = annot_dfB[annot_dfB.description=='saccade']

#Convert 'onset' and 'duration' to seconds if they are in datetime format
if pd.api.types.is_datetime64_any_dtype(saccade_annotationsB['onset']):
    saccade_annotationsB.loc[:, 'onset'] = (saccade_annotationsB['onset'] - annot_dfB.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(saccade_annotationsB['duration']):
    saccade_annotationsB.loc[:, 'duration'] = saccade_annotationsB['duration'] / pd.Timedelta(seconds=1)

saccade_startsB = (saccade_annotationsB['onset'] * cropped_raw_pupilB.info['sfreq']).astype(int)
saccade_endsB = (saccade_startsB + saccade_annotationsB['duration'] * cropped_raw_pupilB.info['sfreq']).astype(int)
print(saccade_startsB, saccade_endsB)
saccade_durationB = np.array(saccade_endsB) - np.array(saccade_startsB)


if pd.api.types.is_datetime64_any_dtype(blink_annotationsB['onset']):
    blink_annotationsB.loc[:, 'onset'] = (blink_annotationsB['onset'] - annot_dfB.iloc[0]['onset'])/pd.Timedelta(seconds=1)

if pd.api.types.is_datetime64_any_dtype(blink_annotationsB['duration']):
    blink_annotationsB.loc[:, 'duration'] = blink_annotationsB['duration'] / pd.Timedelta(seconds=1)

blink_startsB = (blink_annotationsB['onset'] * cropped_raw_pupilB.info['sfreq']).astype(int)
blink_endsB = (blink_startsB + blink_annotationsB['duration'] * cropped_raw_pupilB.info['sfreq']).astype(int)
print(blink_startsB, blink_endsB)
blink_duration = np.array(blink_endsB) - np.array(blink_startsB)


all_saccade_interpB = cropped_raw_pupilB.copy()
#all_saccade_interp = raw_total.crop(tmin=first_target_only,tmax=None)

#% Thresholded Saccade Interpolation

#print(saccade_duration)
print(f"saccade_duration type: {type(saccade_durationB)}, shape: {np.shape(saccade_durationB)}")
for long_duration in saccade_durationB:
    if long_duration >= 83:
        print(f"long saccade duration in samples: {long_duration}")
annot_dfB = all_saccade_interpB.annotations.to_data_frame()
elapsed_timeB= (annot_dfB['onset'] - annot_dfB.iloc[0]['onset'])/pd.Timedelta(seconds=1)

# Print the index and value for events greater than or equal to 0.083
print(len(saccade_annotationsB))

long_saccade_listB = []
for i, long_saccade in saccade_annotationsB['duration'].items():
    if long_saccade >= 0.083:
        print(f"Event: {i}, Duration in seconds: {long_saccade}; and Elapsed time: {elapsed_timeB[i]}")
        long_saccade_listB.append(long_saccade)
# Define the duration threshold for long saccades in samples
duration_threshold = 83

# Calculate the saccade durations in samples
saccade_durationB = np.array(saccade_endsB) - np.array(saccade_startsB)


# Loop through all saccades and apply the window and interpolation only for long saccades
for idx in range(len(saccade_durationB)):

    # Check if the current saccade's duration is greater than or equal to the threshold
    if saccade_durationB[idx] >= duration_threshold:

        # Define window around the saccade (you can adjust this window as needed)
        window = (-100, 100)  # 30 samples before the start and 30 samples after the end, adjust as needed

        start = np.array(saccade_startsB)[idx] + window[0]
        end = np.array(saccade_endsB)[idx] + window[1]

        # Ensure the window stays within bounds of the pupil data
        if start < 0:
            print(f"Saccade #{idx}: Start of window is before data range; setting start to 0.")
            start = 0

        if end > len(pupil_dataB):
            print(f"Saccade #{idx}: End of window is larger than data; reducing window to last datapoint.")
            end = len(pupil_dataB) - 13

        # Define the interpolation range
        interp_points = np.arange(start, end)
        
        anchor = 50
        
        # Define any number of points around the start and end for interpolation, adjust as needed
        start_points_x = np.arange(max(0, start - anchor), start)
        #start_points_x = np.array([start - 12, start - 10, start - 8, start - 6, start - 4, start - 2])
        start_points_x = start_points_x[(start_points_x >= 0) & (start_points_x < len(pupil_dataB))]
        start_points_y = pupil_dataB[start_points_x]

        # Define points after the end
        end_points_x = np.arange(end, min(len(pupil_dataB), end + anchor))
        #end_points_x = np.array([end + 2, end + 4, end + 6, end + 8, end + 10, end + 12])
        end_points_x = end_points_x[(end_points_x >= 0) & (end_points_x < len(pupil_dataB))]
        end_points_y = pupil_dataB[end_points_x]

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
            all_saccade_interpB._data[:, start:start + len(new_section2)] = new_section2
        except Exception as e:
            print(f"Error processing saccade #{idx}: {e}")


print(start, end)
print(window)

#%%



#%%filter part B for annotations
cropped_filtB = all_saccade_interpB.copy().filter(None, 5)
cropped_filtB.pick('pupil_right').plot(title=subj + " Pupil w/ Pchip Interpolation: Part A")

# Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_durationB = cropped_filtB.times[-1]  # Total time in seconds
bad_durationsB = sum([annot['duration'] for annot in cropped_filtB.annotations if annot['description'] == 'BAD_blink'])
percent_badB = (bad_durationsB / total_durationB) * 100
print(percent_badB)


#%% part B: check new annotations and interactive plot with new annotations
dfB = cropped_filtB.annotations.to_data_frame()
dfB[dfB['description']=='BAD_remove'] # or whatever BAD_ has been renamed to

croppedB_annot = all_saccade_interpB.copy().filter(None, 5)
croppedB_annot.set_annotations(cropped_filtB.annotations)

print(len(cropped_filtB.annotations), len(cropped_raw_pupilB.annotations))
croppedB_annot.pick('pupil_right').plot()


#%% part B SAME AS LINE 196 - Gives the percent of a give condition, i.e. BAD_blink, saccade, fixation
total_durationB = cropped_filtB.times[-1]  # Total time in seconds
bad_durationsB = sum([annot['duration'] for annot in cropped_filtB.annotations if annot['description'] == 'bad_remove'])
percent_badB = (bad_durationsB / total_durationB) * 100
print(percent_badB)



#%%
old_dir, old_name = os.path.split(str(croppedB_annot.filenames[0]))
new_name = os.path.join(old_dir, f"{subj}_pupilannot_raw_partB.fif")
print(new_name)

croppedB_annot.save(new_name, overwrite=True)

#%%


