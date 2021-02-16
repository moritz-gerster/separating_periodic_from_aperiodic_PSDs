"""
This script loads fif data after they were annotated with bad segments
and missing channels and saves them as numpy arrays after setting all missing
and bad channels to np.nan.
"""
import os
import mne
import numpy as np

# =============================================================================
# Create MNE Info
# =============================================================================
n_sub = 14
ch_names = ['SMA', 'leftM1', 'rightM1',
            'STN_R01', 'STN_R12', 'STN_R23',
            'STN_L01', 'STN_L12', 'STN_L23',
            'EMG_R', 'EMG_L',
            'HEOG', 'VEOG',
            'event']
sfreq = 2400
ch_types = ["mag", "mag", "mag",
            "seeg", "seeg", "seeg", "seeg", "seeg", "seeg",
            "emg", "emg",
            "eog", "eog",
            "stim"]

info = mne.create_info(ch_names, sfreq, ch_types, verbose=True)
info["highpass"] = 1
info["lowpass"] = 600

plot_options = {
            "duration": 10,  # 10 on notebook screen
            "n_channels": 15,
            "scalings": dict(mag=400, seeg=12, eog=100, emg=500, stim=1),
            "color": dict(mag='darkblue', seeg='black',
                          eog='grey', emg='brown', stim='k'),
            "group_by": "original"
            }
# %%
# =============================================================================
# LOAD Files
# =============================================================================

for cond in ["on", "off"]:

    for subj in range(n_sub):

        path = '../../data/1_raw_clean_fif/rest/subj'
        path_subj = path + f'{subj+1}/{cond}/'  # subjects start at 1
        fnames = os.listdir(path_subj)  # load first file only
        # delete the wird ".DS_Store" files
        real_fnames = [fname for fname in fnames if fname[0] != "."]
        print(f"Subject {subj+1} {cond} has {len(real_fnames)} file(s).")

        raws = []
        for fname in real_fnames:
            raw = mne.io.read_raw_fif(path_subj + fname, info)
            raws.append(raw)
            #  fig = raw.plot(title=fname, **plot_options)
            print(raw.info["bads"])

        # Set bad segments to nan
        bad_segments = [raw.annotations.copy() for raw in raws]

        reject_segments = []
        for bad_seg in bad_segments:
            reject_segments.append([(int(seg["onset"]*sfreq),
                                     int((seg["onset"] +
                                          seg["duration"]) * sfreq))
                                    for seg in bad_seg])

        # Set bad channels to nan
        bad_channels = [raw.info["bads"] for raw in raws]

        reject_channels = []
        for bad_ch in bad_channels:
            reject_channels.append([ch_names.index(bad) for bad in bad_ch])

        # print("Bad segments: ", reject_segments)
        print("Bad channels: ", reject_channels)

        # #### Extract as numpy array
        arrays = []
        for raw in raws:
            time_samples = raw.get_data().shape[1]
            arr = np.zeros([len(ch_names), time_samples])
            for i, ch in enumerate(ch_names):
                try:
                    arr[i] = raw[ch][0]
                # Set missing channels to nan
                except ValueError:
                    arr[i] = np.nan
                    print(f"Subj {subj+1} misses channel {ch}")
            arrays.append(arr)

        for array, reject_seg in zip(arrays, reject_segments):
            # ...would be more efficient if using mask instead of loop
            for reject in reject_seg:
                array[:, reject[0]:reject[1]] = np.nan

        for array, reject_ch in zip(arrays, reject_channels):
            array[reject_ch, :] = np.nan

        path_npy = f"../../data/1_clean_numpy2/rest/subj{subj+1}/{cond}/"

        if not os.path.exists(path_npy):
            os.makedirs(path_npy)

        for arr, fname in zip(arrays, real_fnames):
            np.save(path_npy + fname[:-4] + ".npy", arr)

        print(f"Files saved at\n {path_npy}")
