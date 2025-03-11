import os
import re
import random
import torch
import numpy as np
import pandas as pd


def dataset_split(path= './ETPAD.v2/LIVE_EYE_MOVEMENTS/'):
    subject_ids = sorted({re.search(r'(\d+)_', f).group(1) for f in os.listdir(path) if f.endswith('.txt') and re.search(r'(\d+)_', f)})
    random.seed(42)
    random.shuffle(subject_ids)

    train_size = int(0.6 * len(subject_ids))
    test_size = int(0.2 * len(subject_ids))

    train_subjects = subject_ids[:train_size]
    test_subjects = subject_ids[train_size:train_size + test_size]
    heldout_subjects = subject_ids[train_size + test_size:]

    return train_subjects, test_subjects, heldout_subjects


def velocity_from_raw_datafile(filename, stride, window_size):

    datafile = pd.read_csv(filename, sep="\t")
    data_array = datafile[["X degree", "Y degree", "Data is valid"]].to_numpy()

    invalid_mask = data_array[:, 2] == 1
    data_array[invalid_mask, :] = np.nan
    data_array = data_array[:, :-1]
    
    velocity = (data_array[2:] - data_array[:-2]) / 2
    horizontal_channel, vertical_channel = velocity[:, 0], velocity[:, 1]
    
    horizontal_channel_window = np.lib.stride_tricks.sliding_window_view(horizontal_channel, window_size)[::stride].copy()
    vertical_channel_window = np.lib.stride_tricks.sliding_window_view(vertical_channel, window_size)[::stride].copy()
    
    concatenated_channel = np.stack((horizontal_channel_window, vertical_channel_window), axis=1)
    
    return concatenated_channel

def data_scaling(train_data, test_data):

    mean_ = np.nanmean(train_data, axis=(0, 2), keepdims=True)
    stdv_ = np.nanstd(train_data, axis=(0, 2), keepdims=True)
    
    train_data = (train_data - mean_) / stdv_
    test_data = (test_data - mean_) / stdv_
    
    return train_data, test_data

def data_from_train_subject(stride, window_size, train_subjects, live_path, sasi_path):

    train_live, train_sasi = [], []
    for subject_number in train_subjects:
        for session_number in [1, 2]:
            live_filename = f"{live_path}{subject_number}_{session_number}.txt"
            sasi_filename = f"{sasi_path}{subject_number}_{session_number}.txt"
            train_live.append(velocity_from_raw_datafile(live_filename, stride, window_size))
            train_sasi.append(velocity_from_raw_datafile(sasi_filename, stride, window_size))

    train_live, train_sasi = np.concatenate(train_live, axis=0), np.concatenate(train_sasi, axis=0)
    train_live_label = np.ones(train_live.shape[0])
    train_sasi_label = np.zeros(train_sasi.shape[0])
    
    train_data = np.concatenate((train_live, train_sasi), axis=0)
    train_labels = np.concatenate((train_live_label, train_sasi_label), axis=0)

    return train_data, train_labels

def data_from_test_subject(stride, window_size, test_subjects, live_path, sasi_path):

    test_live, test_sasi = [], []
    for subject_number in test_subjects:
        for session_number in [1, 2]:
            live_filename = f"{live_path}{subject_number}_{session_number}.txt"
            sasi_filename = f"{sasi_path}{subject_number}_{session_number}.txt"
            test_live.append(velocity_from_raw_datafile(live_filename, stride, window_size))
            test_sasi.append(velocity_from_raw_datafile(sasi_filename, stride, window_size))

    test_live, test_sasi = np.concatenate(test_live, axis=0), np.concatenate(test_sasi, axis=0)
    test_live_label = np.ones(test_live.shape[0])
    test_sasi_label = np.zeros(test_sasi.shape[0])
    
    test_data = np.concatenate((test_live, test_sasi), axis=0)
    test_labels = np.concatenate((test_live_label, test_sasi_label), axis=0)

    return test_data, test_labels

def train_test_split(stride, window_size, train_subjects, test_subjects, live_path, sasi_path):

    train_data, train_labels = data_from_train_subject(stride, window_size, train_subjects, live_path, sasi_path)
    test_data, test_labels = data_from_test_subject(stride, window_size, test_subjects, live_path, sasi_path)
    
    train_data, test_data = data_scaling(train_data, test_data)

    X_train = torch.tensor(train_data, dtype=torch.float)
    y_train = torch.tensor(train_labels, dtype=torch.float)
    X_test = torch.tensor(test_data, dtype=torch.float)
    y_test = torch.tensor(test_labels, dtype=torch.float)

    # Handle NaN values
    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_test = torch.nan_to_num(X_test, nan=0.0)

    return X_train, y_train, X_test, y_test

def data_from_heldout_subject(stride, window_size, heldout_subjects, live_path, sasi_path):

    test_live, test_sasi = [], []
    for subject_number in heldout_subjects:
        for session_number in [1, 2]:
            live_filename = f"{live_path}{subject_number}_{session_number}.txt"
            sasi_filename = f"{sasi_path}{subject_number}_{session_number}.txt"
            test_live.append(velocity_from_raw_datafile(live_filename, stride, window_size))
            test_sasi.append(velocity_from_raw_datafile(sasi_filename, stride, window_size))

    test_live, test_sasi = np.concatenate(test_live, axis=0), np.concatenate(test_sasi, axis=0)
    test_live_label = np.ones(test_live.shape[0])
    test_sasi_label = np.zeros(test_sasi.shape[0])
    
    test_data = np.concatenate((test_live, test_sasi), axis=0)
    test_labels = np.concatenate((test_live_label, test_sasi_label), axis=0)

    return test_data, test_labels

def heldout_split(stride, window_size, train_subjects, heldout_subjects, live_path, sasi_path):

    train_data, _ = data_from_train_subject(stride, window_size, train_subjects, live_path, sasi_path)
    test_data, test_labels = data_from_heldout_subject(stride, window_size, heldout_subjects, live_path, sasi_path)
    
    train_data, test_data = data_scaling(train_data, test_data)

    X_test = torch.tensor(test_data, dtype=torch.float)
    y_test = torch.tensor(test_labels, dtype=torch.float)

    X_test = torch.nan_to_num(X_test, nan=0.0)

    return X_test, y_test

