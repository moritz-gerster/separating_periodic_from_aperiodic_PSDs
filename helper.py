"""Helper functions."""
import os
import mne
import numpy as np

conditions = ["on", "off"]
n_sub = 14
task = "rest"

# Paths
load_folder = "../../Hypo1-1/2021-03-03_Wrap_Up/"
fif_path = f'../../Hypo1-1/data/1_raw_clean_fif/{task}/'
psd_path = load_folder + f"data/{task}/PSD_arrays/"

# Names
psd_name = "_psd.npy"
freqs_name = "freqs.npy"


def load_fif():
    """
    Load fif data without bad segments, return as list.

    Parameters
    ----------
    fif_path : string
        Where fif data is stored.

    Returns
    -------
    raw_on: list of all subjects during on condition
    raw_off: list of all subjects during off condition

    """
    raw_conds = []

    # drop channels
    chs_delete = ['EMG_R', 'EMG_L', 'HEOG', 'VEOG', 'event']

    for cond in conditions:
        raw_cond = []
        for subj in range(n_sub):

            path_subj = fif_path + f'subj{subj+1}/{cond}/'
            # list files
            files = os.listdir(path_subj)
            # filter out bad files
            good_files = [file for file in files if file.startswith('subj')]
            # if more than 1 good file choose the first
            fname = good_files[0]
            data_subj = mne.io.read_raw_fif(path_subj + fname,
                                            preload=True)
            for delete in chs_delete:
                try:
                    data_subj.drop_channels(delete)
                except ValueError:  # if channels are missing
                    continue
            raw_cond.append(data_subj)
        raw_conds.append(raw_cond)
    return raw_conds


def load_psd():
    """
    Load data needed for this script.

    Returns
    -------
    freqs : TYPE
        DESCRIPTION.
    psd_on : TYPE
        DESCRIPTION.
    psd_off : TYPE
        DESCRIPTION.

    """
# =============================================================================
#     # CREATE SAVE FOLDER
#     Path(fig_spectra_path).mkdir(parents=True, exist_ok=True)
# =============================================================================
    # load coherencies and frequencies
    freqs = np.load(psd_path + freqs_name)
    psd_on = np.load(psd_path + "on" + psd_name)
    psd_off = np.load(psd_path + "off" + psd_name)
    return freqs, psd_on, psd_off