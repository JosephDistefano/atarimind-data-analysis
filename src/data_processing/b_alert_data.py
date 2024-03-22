import os

import mne
from autoreject import get_rejection_threshold

from .extract_data import read_xdf_eeg_data
from .mne_write_edf import write_edf


def decontaminate_eeg(raw_eeg, config, ica_clean):
    # Drop auxillary channels
    try:
        raw_eeg = raw_eeg.drop_channels(['ECG', 'AUX1', 'AUX2', 'AUX3'])
    except ValueError:
        pass

    # Filtering
    raw_eeg.notch_filter(60, filter_length='auto', phase='zero',
                         verbose=False)  # Line noise
    raw_eeg.filter(l_freq=0.5, h_freq=50, fir_design='firwin',
                   verbose=False)  # Band pass filter

    # Channel information
    raw_eeg.set_montage(montage="standard_1020", verbose=False)
    ch_info = {
        'Fp1': 'eeg',
        'F7': 'eeg',
        'F8': 'eeg',
        'T4': 'eeg',
        'T6': 'eeg',
        'T5': 'eeg',
        'T3': 'eeg',
        'Fp2': 'eeg',
        'O1': 'eeg',
        'P3': 'eeg',
        'Pz': 'eeg',
        'F3': 'eeg',
        'Fz': 'eeg',
        'F4': 'eeg',
        'C4': 'eeg',
        'P4': 'eeg',
        'POz': 'eeg',
        'C3': 'eeg',
        'Cz': 'eeg',
        'O2': 'eeg'
    }
    raw_eeg.set_channel_types(ch_info)

    # Epoch the data and get the global rejection threshold
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(raw_eeg, duration=epoch_length)
    epochs = mne.Epochs(raw_eeg,
                        events,
                        picks=['eeg'],
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)

    # Get the rejection and flat threshold using autoreject
    reject_threshold = get_rejection_threshold(epochs.load_data(),
                                               random_state=42)
    flat_threshold = dict(eeg=1e-6)

    # Drop the bad amplitude segments
    epochs.drop_bad(reject=reject_threshold, flat=flat_threshold)

    # Convert the data to mne Raw format
    data = epochs.get_data().transpose(1, 0, 2).reshape(20, -1)
    info = epochs.info
    decon_eeg = mne.io.RawArray(data, info)
    return decon_eeg


def write_mne_to_b_alert_edf(config, clean_with_ica, save_data):
    """This functions writes the mne epoch into b-alert redable .edf format
    Parameters
    ----------
    config : yaml
        The configuration file
    save_data : bool
        Whether to save the data or not
    """
    for subject in config['subjects']:
        for session in config['sessions']:
            # Read the raw
            raw_eeg, time_stamps = read_xdf_eeg_data(config, subject, session)

            # Decontaminate the EEG files
            decon_eeg = decontaminate_eeg(raw_eeg, config, clean_with_ica)

            # Save the file
            subject_file = 'sub-OFS_' + subject
            session_file = 'ses-' + session
            edf_file = ''.join(
                [subject, '11000_ses-', session, '_task-T1_run-001.edf'])
            save_path = ''.join([
                config['raw_xdf_path'], subject_file, '/', session_file,
                '/b-alert/'
            ])

            # Make the directory if not present
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if save_data:
                write_edf(decon_eeg, save_path + edf_file, overwrite=True)
    return None
