import ujson
import csv
import os
import pyxdf
import numpy as np
import mne
import pickle
import pandas as pd
import csv
import numpy
import datetime
from scipy.stats import zscore
import statistics
from itertools import groupby
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from autoreject import get_rejection_threshold, AutoReject, compute_thresholds
from .mne_import_xdf import read_raw_xdf
from .mne_write_edf import write_edf

def read_xdf_eeg_data(config,file,game,sub,session):

    xdf_file  = file 
    read_path = config['raw_data_path'] + game + '/' + xdf_file
    path = Path(read_path)
    if path.is_file():
        raw_eeg, time_stamps = read_raw_xdf(read_path)
        if raw_eeg != []:
            raw_eeg = raw_eeg.drop_channels(["Offset","Hour","Min","Sec","MilliSec","AUX1","AUX2","AUX3","Epoch"])
            flag = True
            flag_2 = True
    else:
        flag = False
        raw_eeg = []
        time_stamps =[]
        flag_2 = False
    if raw_eeg == []:
        flag = False
        time_stamps = []
        flag = False
        flag_2 = False
    return raw_eeg, time_stamps, flag

def write_mne_to_b_alert_edf(config, save_data):
    b_alert_names = config['b_alert_subjects']

    for game in config['games']:
        x = 0

        read_path = config['raw_data_path'] + game + '/' 
        files = os.listdir(read_path)
        for file in files:
            sub = file[4:6]
            session = file[-15:-8]
            b_alert_name = b_alert_names[x]         
            raw_eeg, time_stamps,flag = read_xdf_eeg_data(config,file,game,sub,session)
            if flag:
                subject_file = sub
                session_file = session
                edf_file = "".join([b_alert_name, "00000_",session ,".edf"])
                save_path = config['raw_edf_path'] + game + '/' + edf_file
                # Make the directory if not present
                # if not os.path.isdir(save_path):
                #     os.mkdir(save_path)
                if save_data:
                    write_edf(raw_eeg, save_path, overwrite=True)
        x = x+1
    return None

def clean_with_ica(raw_eeg,  show_ica=False):
    """Clean epochs with ICA.
    Parameter
    ----------
    epochs : Filtered raw EEG
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs
    """
    picks = mne.pick_types(
        raw_eeg.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
    )
    ica = mne.preprocessing.ICA(n_components=None, method="picard", verbose=False)
    # Epoch the data and get the global rejection threshold
    epoch_length = 2
    events = mne.make_fixed_length_events(raw_eeg, duration=epoch_length)
    epochs = mne.Epochs(
        raw_eeg,
        events,
        picks=["eeg"],
        tmin=0,
        tmax=2,
        baseline=(0, 0),
        verbose=False,
    )
    # Get the rejection threshold using autoreject
    reject_threshold = compute_thresholds(
        epochs.load_data(), method="bayesian_optimization", random_state=42, n_jobs=10
    )
    ica.fit(epochs, picks=picks, reject=reject_threshold, tstep=epoch_length)
    # Extra caution to detect the eye blinks
    # ica = append_eog_index(epochs, ica)  # Append the eog index to ICA
    # mne pipeline to detect artifacts
    ica.detect_artifacts(epochs, eog_criterion=range(2))
    if show_ica:
        ica.plot_components(inst=epochs)
    cleaned_eeg = ica.apply(raw_eeg)  # Apply the ICA on raw EEG
    return cleaned_eeg, ica

def append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA
    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended
    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs, ch_name="Fp1", verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.85]
    ica.exclude += id_eog
    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs, ch_name="Fp2", verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.85]
    ica.exclude += id_eog
    return ica

def decontaminate_eeg(raw_eeg, ica_clean):
    # Drop auxillary channels
    try:
        raw_eeg = raw_eeg.drop_channels(
            [
                "Epoch",
                "Offset",
                "Hour",
                "Min",
                "Sec",
                "MilliSec",
                "AUX1",
                "AUX2",
                "AUX3",
                "ECG"
            ]
        )
    except ValueError:
        pass
    ch_info = {
        "Fp1": "eeg",
        "F7": "eeg",
        "F8": "eeg",
        "T4": "eeg",
        "T6": "eeg",
        "T5": "eeg",
        "T3": "eeg",
        "Fp2": "eeg",
        "O1": "eeg",
        "P3": "eeg",
        "Pz": "eeg",
        "F3": "eeg",
        "Fz": "eeg",
        "F4": "eeg",
        "C4": "eeg",
        "P4": "eeg",
        "POz": "eeg",
        "C3": "eeg",
        "Cz": "eeg",
        "O2": "eeg",
    }
    raw_eeg.set_channel_types(ch_info)
    # Filtering
    raw_eeg.notch_filter(
        60, filter_length="auto", phase="zero", verbose=False
    )  # Line noise
    raw_eeg.filter(
        l_freq=0.5, h_freq=50, fir_design="firwin", verbose=False
    )  # Band pass filter
    # Channel information
    raw_eeg.set_montage(montage="standard_1020")  # , set_dig=True, verbose=False)
    # Epoch the data and get the global rejection threshold
    epoch_length = 1
    events = mne.make_fixed_length_events(raw_eeg, duration=epoch_length)
    epochs = mne.Epochs(
        raw_eeg,
        events,
        picks=["eeg"],
        tmin=0,
        tmax=3,
        baseline=(0, 0),
        verbose=False,
    )
    # Get the rejection and flat threshold using autoreject
    reject_threshold = get_rejection_threshold(epochs.load_data(), random_state=42)
    flat_threshold = dict(eeg=1e-6)
    if ica_clean:
        decon_eeg, _ = clean_with_ica(raw_eeg, show_ica=False)
    else:
        # Drop the bad amplitude segments
        epochs.drop_bad(reject=reject_threshold, flat=flat_threshold)
        # Convert the data to mne Raw format
        data = epochs.get_data().transpose(1, 0, 2).reshape(20, -1)
        info = epochs.info
        decon_eeg = mne.io.RawArray(data, info)
    return decon_eeg

