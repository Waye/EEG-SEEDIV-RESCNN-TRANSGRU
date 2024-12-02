##############
## Extract DE features for 5 frequency bands from each channel of the SEED dataset,
## and convert the 62-channel data into an 8*9*5 three-dimensional input,
## where 8*9 represents the 2D plane after converting the 62 channels, and 5 represents the 5 frequency bands
##############

import os
import sys
import math
import numpy as np
# import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat

# Function to decompose EEG data into different frequency bands and extract DE features
# Input: file path, name (shortened name of the participant)
def decompose(file, name):
    # Load the .mat file containing the EEG data
    data = loadmat(file)
    frequency = 200  # Sampling rate of the SEED dataset is downsampled to 200Hz

    # Create an empty array to store DE features and labels
    decomposed_de = np.empty([0, 62, 5])
    label = np.array([])
    all_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3,
             2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1,
             1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0] # Labels for each trial

    # Loop through all 24 trials in the dataset
    for trial in range(24):
        # Load the EEG data for the current trial
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        # Number of samples per segment (0.5 seconds per segment, with a sampling rate of 200Hz)
        num_sample = int(len(tmp_trial_signal[0]) / 100)
        print('{}-{}'.format(trial + 1, num_sample))

        # Initialize temporary array to store DE features for each channel
        temp_de = np.empty([0, num_sample])
        # Assign labels for each sample in the current trial
        label = np.append(label, [all_label[trial]] * num_sample)

        # Loop through each channel (total 62 channels)
        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]

            # Apply bandpass filters to extract different frequency bands
            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

            # Initialize arrays to store DE values for each frequency band
            DE_delta = np.zeros(shape=[0], dtype=float)
            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            # Compute DE features for each frequency band in each segment
            for index in range(num_sample):
                DE_delta = np.append(DE_delta, compute_DE(delta[index * 100:(index + 1) * 100]))
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 100:(index + 1) * 100]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 100:(index + 1) * 100]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 100:(index + 1) * 100]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 100:(index + 1) * 100]))
            
            # Stack the DE features for each frequency band
            temp_de = np.vstack([temp_de, DE_delta])
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        # Reshape the DE features to match the desired format
        temp_trial_de = temp_de.reshape(-1, 5, num_sample)
        print("temp_trial_de:", temp_trial_de.shape)  # Print the shape of the reshaped DE features
        temp_trial_de = temp_trial_de.transpose([2, 0, 1])  # Rearrange dimensions to match desired format
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])  # Stack trial data

    print("trial_DE shape:", decomposed_de.shape)
    return decomposed_de, label

# Function to create bandpass filter coefficients
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply bandpass filter to data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to compute Differential Entropy (DE) of a signal
def compute_DE(signal):
    variance = np.var(signal, ddof=1)  # Compute variance of the signal
    return math.log(2 * math.pi * math.e * variance) / 2  # Return DE value

# Main script to extract features and save data
import os
import numpy as np

file_path = 'D:/BigData/SEED_IV/SEED_IV/eeg_raw_data/'

# List of participant names and short names used in file naming
people_name = ['1_20160518', '1_20161125','1_20161126',
               '2_20150915', '2_20150920','2_20151012',
               '3_20150919','3_20151018','3_20151101',
               '4_20151111', '4_20151118','4_20151123',
               '5_20160406', '5_20160413','5_20160420',
               '6_20150507','6_20150511','6_20150512',
               '7_20150715', '7_20150717','7_20150721',
               '8_20151103', '8_20151110','8_20151117',
               '9_20151028','9_20151119','9_20151209',
               '10_20151014', '10_20151021','10_20151023',
               '11_20150916', '11_20150921','11_20151011',
               '12_20150725','12_20150804','12_20150807',
               '13_20151115', '13_20151125','13_20161130',
               '14_20151205', '14_20151208','14_20151215',
               '15_20150508','15_20150514','15_20150527']

short_name = ['cz', 'cz','cz',
              'ha', 'ha', 'ha',
              'hql','hql','hql',
              'ldy','ldy','ldy',
              'ly','ly','ly',
              'mhw','mhw','mhw',
              'mz','mz','mz',
              'qyt','qyt','qyt',
              'rx','rx','rx',
              'tyc','tyc','tyc',
              'whh','whh','whh',
              'wll','wll','wll',
              'wq','wq','wq',
              'zjd','zjd','zjd',
              'zjy','zjy','zjy']

# Initialize empty arrays for storing the final DE features and labels
X = np.empty([0, 62, 5])
y = np.empty([0, 1])

# Loop through all participants to extract DE features
for i in range(len(people_name)):  # Loop through all 45 experiments (15 participants, 3 trials each)
    file_name = file_path + people_name[i]
    print('processing {}'.format(people_name[i]))
    decomposed_de, label = decompose(file_name, short_name[i])  # Extract DE features for each participant
    X = np.vstack([X, decomposed_de])  # Stack the features for all participants
    y = np.append(y, label)  # Stack the labels for all participants

# Save the extracted DE features and labels as .npy files
np.save("D:/BigData/SEED_IV/SEED_IV/DE0.5s/X_1D.npy", X)
np.save("D:/BigData/SEED_IV/SEED_IV/DE0.5s/y.npy", y)

# Load the saved features and labels
X = np.load('D:/BigData/SEED_IV/SEED_IV/DE0.5s/X_1D.npy')
y = np.load('D:/BigData/SEED_IV/SEED_IV/DE0.5s/y.npy')

# Reshape the 62-channel data into an 8x9 matrix (based on the 10-20 electrode system)
X89 = np.zeros((len(y), 8, 9, 5))  # Create an empty array for the 8x9x5 data
X89[:, 0, 2, :] = X[:, 3, :]  # Assign values to specific positions in the 8x9 matrix
X89[:, 0, 3:6, :] = X[:, 0:3, :]
X89[:, 0, 6, :] = X[:, 4, :]

# Assign values for the middle rows of the 8x9 matrix
for i in range(5):
    X89[:, i + 1, :, :] = X[:, 5 + i * 9:5 + (i + 1) * 9, :]

# Assign values for the last two rows of the 8x9 matrix
X89[:, 6, 1:8, :] = X[:, 50:57, :]
X89[:, 7, 2:7, :] = X[:, 57:62, :]

# Save the reshaped 8x9 matrix data
np.save("D:/BigData/SEED_IV/SEED_IV/DE0.5s/X89.npy", X89)
