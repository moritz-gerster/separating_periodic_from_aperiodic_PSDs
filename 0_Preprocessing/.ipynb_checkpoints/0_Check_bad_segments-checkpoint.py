import os
import mne
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Create MNE Info
# =============================================================================
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
            "duration": 10, # 10 on notebook screen
            "n_channels": 13,
            "scalings": dict(mag=400, seeg=12, eog=100, emg=500, stim=1),
            "color": dict(mag='darkblue', seeg='black',
                          eog='grey', emg='brown',stim='k'),
            "group_by": "original"
            }
# %%
# =============================================================================
# LOAD Files
# =============================================================================
subj = 5
cond = "on"
# =============================================================================
# =============================================================================
path = '../../data/raw_annotated_fif/rest/subj'
path_subj = path + f'{subj+1}/{cond}/'  # subjects start at 1
fnames = os.listdir(path_subj)  # load first file only
# delete the wird ".DS_Store" files
real_fnames = [fname for fname in fnames if fname[0] != "."] 
print(f"Subject {subj+1} {cond} has {len(real_fnames)} file(s).")

raws = []
for fname in real_fnames:
    raws.append(mne.io.read_raw_fif(path_subj + fname, info))

for i, raw in enumerate(raws):
    fig = raw.plot(title=real_fnames[i], **plot_options)

# %%
# =============================================================================
# # make sure to mark the bad segments before continueing the script
1/0
# =============================================================================




# %%
# =============================================================================
# # Check Raw data of all channels for each subject entire recording
# =============================================================================

bad_segments = [raw.annotations.copy() for raw in raws]

reject_indices = []
for bad_segs in bad_segments:
    reject_indices.append([(int(seg["onset"]*sfreq),
                            int((seg["onset"] + seg["duration"])*sfreq))
                           for seg in bad_segs])

print(f"Bad segments: ", reject_indices)

# =============================================================================
# #### Extract as numpy array
# =============================================================================
arrays = [arr.get_data() for arr in raws]

for array, reject_index in zip(arrays, reject_indices):
    for reject in reject_index:
        array[:, reject[0]:reject[1]] = np.nan


# =============================================================================
# # Save as numpy and fif
# =============================================================================
path_fif = f"../../data/raw_annotated_fif/rest/subj{subj+1}/{cond}/"

if not os.path.exists(path_fif):
    os.makedirs(path_fif)

for raw, fname in zip(raws, real_fnames):
    raw.save(path_fif + fname[:-4] + "_raw.fif")

path_npy = f"../../data/numpy_bad_seg_is_nan/rest/subj{subj+1}/{cond}/"

if not os.path.exists(path_npy):
    os.makedirs(path_npy)

for arr, fname in zip(arrays, real_fnames):
    np.save(path_npy + fname[:-4] + "_raw.npy", arr)

print(f"Files saved at\n {path_fif}\n and\n {path_npy}")
for fig in raws:
    plt.close()

# =============================================================================
# ### Optional:
# ### ... test if annotations are correct:
#
# test = mne.io.read_raw_fif(path_fif + fname[:-4] + "_raw.fif", info)
# _ = test.plot(title=fname[:-4], **plot_options)
# =============================================================================
